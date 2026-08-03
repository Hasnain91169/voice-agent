"""Tests for rechunking inbound audio into fixed 20 ms frames."""

from __future__ import annotations

from tests.conftest import square_pcm
from voice_agent.audio import wav
from voice_agent.audio.framing import Rechunker, slice_frames
from voice_agent.config import BYTES_PER_FRAME, FRAME_MS


class TestRechunker:
    def test_exact_frame_emits_one(self) -> None:
        frames = Rechunker().push(square_pcm(500))
        assert len(frames) == 1
        assert len(frames[0].pcm) == BYTES_PER_FRAME

    def test_partial_input_emits_nothing_and_is_retained(self) -> None:
        rechunker = Rechunker()
        assert rechunker.push(b"\x00" * (BYTES_PER_FRAME - 2)) == []
        assert rechunker.pending_bytes == BYTES_PER_FRAME - 2

    def test_two_ten_ms_slices_make_one_frame(self) -> None:
        # This is exactly what Vonage does, and the reason this class exists.
        rechunker = Rechunker()
        half = BYTES_PER_FRAME // 2
        assert rechunker.push(b"\x11" * half) == []
        assert len(rechunker.push(b"\x11" * half)) == 1

    def test_remainder_carries_into_the_next_push(self) -> None:
        rechunker = Rechunker()
        frames = rechunker.push(b"\x00" * (BYTES_PER_FRAME * 2 + 100))
        assert len(frames) == 2
        assert rechunker.pending_bytes == 100

        frames = rechunker.push(b"\x00" * (BYTES_PER_FRAME - 100))
        assert len(frames) == 1
        assert rechunker.pending_bytes == 0

    def test_sequence_numbers_increment_across_pushes(self) -> None:
        rechunker = Rechunker()
        first = rechunker.push(square_pcm(100, frames=2))
        second = rechunker.push(square_pcm(100, frames=1))
        assert [f.seq for f in first] == [0, 1]
        assert [f.seq for f in second] == [2]

    def test_energy_is_computed_at_rechunk_time(self) -> None:
        # Both the utterance VAD and the barge-in detector read this; computing
        # it twice on the hot path would be wasteful.
        frame = Rechunker().push(square_pcm(1_234))[0]
        assert frame.rms == 1_234.0

    def test_empty_push_is_a_no_op(self) -> None:
        rechunker = Rechunker()
        assert rechunker.push(b"") == []
        assert rechunker.pending_bytes == 0

    def test_reset_drops_partial_but_keeps_sequence(self) -> None:
        rechunker = Rechunker()
        rechunker.push(square_pcm(100))
        rechunker.push(b"\x00" * 200)
        rechunker.reset()
        assert rechunker.pending_bytes == 0
        assert rechunker.frames_emitted == 1

    def test_frame_duration_is_the_configured_frame(self) -> None:
        assert Rechunker().push(square_pcm(100))[0].duration_ms == FRAME_MS

    def test_endian_swap_changes_interpretation(self) -> None:
        raw = square_pcm(256)
        plain = Rechunker().push(raw)[0]
        swapped = Rechunker(swap_endian=True).push(raw)[0]
        assert plain.pcm != swapped.pcm
        assert swapped.pcm == wav.swap_endian(raw)


class TestSliceFrames:
    def test_splits_into_whole_frames(self) -> None:
        assert len(list(slice_frames(square_pcm(100, frames=5)))) == 5

    def test_discards_the_remainder(self) -> None:
        pcm = square_pcm(100, frames=2) + b"\x00" * 40
        assert len(list(slice_frames(pcm))) == 2

    def test_shorter_than_one_frame_yields_nothing(self) -> None:
        assert list(slice_frames(b"\x00" * 10)) == []
