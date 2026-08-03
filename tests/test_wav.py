"""Tests for the numpy PCM helpers that replaced audioop."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import square_pcm
from voice_agent.audio import wav
from voice_agent.config import SAMPLE_RATE, SAMPLE_WIDTH


class TestRms:
    def test_silence_is_zero(self) -> None:
        assert wav.rms(b"\x00\x00" * 320) == 0.0

    def test_empty_is_zero_not_an_error(self) -> None:
        # Empty buffers arrive whenever a transport sends a keepalive.
        assert wav.rms(b"") == 0.0

    @pytest.mark.parametrize("amplitude", [1, 100, 5_000, 32_767])
    def test_square_wave_rms_equals_amplitude(self, amplitude: int) -> None:
        assert wav.rms(square_pcm(amplitude)) == pytest.approx(amplitude)

    def test_does_not_overflow_at_full_scale(self) -> None:
        # int16 squared overflows int16; the computation must widen first.
        assert wav.rms(square_pcm(32_767)) == pytest.approx(32_767)


class TestToArray:
    def test_drops_trailing_odd_byte(self) -> None:
        assert wav.to_array(b"\x01\x02\x03").size == 1

    def test_empty_input(self) -> None:
        assert wav.to_array(b"").size == 0


class TestFloatConversion:
    def test_round_trip_preserves_samples(self) -> None:
        original = square_pcm(12_345)
        assert wav.from_float32(wav.to_float32(original)) == original

    def test_normalises_into_unit_range(self) -> None:
        floats = wav.to_float32(square_pcm(32_767))
        assert float(np.abs(floats).max()) < 1.0

    def test_clips_rather_than_wrapping(self) -> None:
        # Wrapping would turn a loud sample into a loud sample of opposite sign,
        # which sounds like a click rather than clean saturation.
        out = wav.to_array(wav.from_float32(np.array([2.0, -2.0], dtype=np.float32)))
        assert out[0] > 0 and out[1] < 0


class TestResample:
    def test_same_rate_is_identity(self) -> None:
        pcm = square_pcm(1_000)
        assert wav.resample(pcm, SAMPLE_RATE, SAMPLE_RATE) is pcm

    def test_downsample_length_ratio(self) -> None:
        # Piper's native 22.05 kHz down to our 16 kHz is the real-world case.
        pcm = square_pcm(1_000, frames=10)
        out = wav.resample(pcm, 22_050, 16_000)
        expected = round((len(pcm) // SAMPLE_WIDTH) * 16_000 / 22_050)
        assert len(out) // SAMPLE_WIDTH == expected

    def test_upsample_length_ratio(self) -> None:
        pcm = square_pcm(1_000, frames=4)
        out = wav.resample(pcm, 8_000, 16_000)
        assert len(out) // SAMPLE_WIDTH == pytest.approx((len(pcm) // SAMPLE_WIDTH) * 2, abs=1)

    def test_preserves_a_constant_signal(self) -> None:
        # A DC level must survive resampling; interpolation between equal
        # neighbours cannot invent variation.
        pcm = np.full(1_000, 4_242, dtype="<i2").tobytes()
        out = wav.to_array(wav.resample(pcm, 22_050, 16_000))
        assert np.allclose(out, 4_242, atol=1)

    def test_empty_input(self) -> None:
        assert wav.resample(b"", 22_050, 16_000) == b""

    def test_rejects_nonsense_rates(self) -> None:
        with pytest.raises(ValueError):
            wav.resample(square_pcm(100), 0, 16_000)


class TestToMono:
    def test_mono_passthrough_is_untouched(self) -> None:
        pcm = square_pcm(500)
        assert wav.to_mono(pcm, 1) is pcm

    def test_averages_stereo_channels(self) -> None:
        interleaved = np.array([100, 300, -200, 0], dtype="<i2").tobytes()
        assert list(wav.to_array(wav.to_mono(interleaved, 2))) == [200, -100]

    def test_rejects_zero_channels(self) -> None:
        with pytest.raises(ValueError):
            wav.to_mono(square_pcm(100), 0)


class TestWavContainer:
    def test_round_trip_at_canonical_format(self) -> None:
        pcm = square_pcm(9_000, frames=3)
        assert wav.decode_wav(wav.encode_wav(pcm)) == pcm

    def test_decode_normalises_rate_and_channels(self) -> None:
        # What an ElevenLabs or Piper response actually looks like: wrong rate,
        # sometimes stereo. Playback only speaks one format.
        stereo = np.repeat(wav.to_array(square_pcm(1_000, frames=5)), 2).tobytes()
        import io
        import wave as wave_mod

        buffer = io.BytesIO()
        with wave_mod.open(buffer, "wb") as writer:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(22_050)
            writer.writeframes(stereo)

        decoded = wav.decode_wav(buffer.getvalue())
        expected_samples = round(5 * 320 * 16_000 / 22_050)
        assert len(decoded) // SAMPLE_WIDTH == pytest.approx(expected_samples, abs=2)

    def test_decodes_8_bit_source(self) -> None:
        import io
        import wave as wave_mod

        buffer = io.BytesIO()
        with wave_mod.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(1)
            writer.setframerate(16_000)
            writer.writeframes(bytes([128, 200, 56, 128]))

        decoded = wav.to_array(wav.decode_wav(buffer.getvalue()))
        assert decoded[0] == 0  # 128 is the unsigned midpoint
        assert decoded[1] > 0 and decoded[2] < 0


class TestStreamingResampler:
    """The seam-free resampler used on Piper's streamed raw output."""

    def test_matches_whole_buffer_resampling(self) -> None:
        # The property that matters: feeding a stream in pieces must produce the
        # same audio as resampling it all at once. Anything else is a seam.
        source = (np.sin(np.arange(4_000) * 0.02) * 8_000).round().astype("<i2").tobytes()
        expected = wav.resample(source, 22_050, 16_000)

        resampler = wav.StreamingResampler(22_050, 16_000)
        chunks = [resampler.process(source[i : i + 512]) for i in range(0, len(source), 512)]
        chunks.append(resampler.flush())
        actual = b"".join(chunks)

        expected_samples = wav.to_array(expected).astype(np.float64)
        actual_samples = wav.to_array(actual).astype(np.float64)
        shared = min(len(expected_samples), len(actual_samples))
        assert abs(len(expected_samples) - len(actual_samples)) <= 2
        # Sample-for-sample agreement, not merely similar length.
        assert np.max(np.abs(expected_samples[:shared] - actual_samples[:shared])) <= 1.0

    def test_no_discontinuity_at_chunk_boundaries(self) -> None:
        # Resampling each chunk independently restarts the interpolator, which
        # is audible as a click every few milliseconds.
        source = (np.sin(np.arange(3_000) * 0.05) * 6_000).round().astype("<i2").tobytes()
        resampler = wav.StreamingResampler(22_050, 16_000)
        out = b"".join(resampler.process(source[i : i + 256]) for i in range(0, len(source), 256))
        samples = wav.to_array(out).astype(np.float64)
        # A smooth sine resampled correctly has small sample-to-sample steps; a
        # seam shows up as an isolated jump far outside the normal range.
        steps = np.abs(np.diff(samples))
        assert steps.max() < steps.mean() * 12

    def test_passthrough_when_rates_match(self) -> None:
        resampler = wav.StreamingResampler(16_000, 16_000)
        assert resampler.passthrough is True
        pcm = square_pcm(1_000)
        assert resampler.process(pcm) is pcm
        assert resampler.flush() == b""

    def test_correct_even_one_sample_at_a_time(self) -> None:
        # The pathological chunking: if the resampler ever consumed a sample it
        # still needed, this is where it would show.
        source = (np.sin(np.arange(600) * 0.07) * 5_000).round().astype("<i2").tobytes()
        expected = wav.to_array(wav.resample(source, 22_050, 16_000)).astype(np.float64)

        resampler = wav.StreamingResampler(22_050, 16_000)
        pieces = [resampler.process(source[i : i + 2]) for i in range(0, len(source), 2)]
        pieces.append(resampler.flush())
        actual = wav.to_array(b"".join(pieces)).astype(np.float64)

        shared = min(len(expected), len(actual))
        assert abs(len(expected) - len(actual)) <= 2
        assert np.max(np.abs(expected[:shared] - actual[:shared])) <= 1.0

    def test_empty_chunks_are_harmless(self) -> None:
        resampler = wav.StreamingResampler(22_050, 16_000)
        assert resampler.process(b"") == b""
        assert resampler.flush() == b""

    def test_rejects_nonsense_rates(self) -> None:
        with pytest.raises(ValueError):
            wav.StreamingResampler(0, 16_000)


class TestSilence:
    def test_duration_maps_to_byte_count(self) -> None:
        assert len(wav.silence(100)) == SAMPLE_RATE // 10 * SAMPLE_WIDTH

    def test_non_positive_is_empty(self) -> None:
        assert wav.silence(0) == b""
        assert wav.silence(-50) == b""

    def test_is_actually_silent(self) -> None:
        assert wav.rms(wav.silence(20)) == 0.0
