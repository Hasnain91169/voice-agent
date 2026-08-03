"""PCM and WAV helpers, backed by numpy.

Deliberately free of :mod:`audioop`. That module was removed in Python 3.13, and
the codebase this replaces carried a try/except import chain reaching for the
``audioop-lts`` backport to stay alive. Since every operation needed here is a
few lines of numpy, the dependency buys nothing and costs a version ceiling.

All functions speak 16-bit signed little-endian PCM as :class:`bytes`, which is
what both transports deliver and what the ASR providers want.
"""

from __future__ import annotations

import io
import wave

import numpy as np

from voice_agent.config import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH

_INT16_MAX = 32_768.0


def to_array(pcm: bytes) -> np.ndarray:
    """View raw PCM as int16 samples.

    A trailing odd byte is dropped rather than raising: partial frames are a
    normal consequence of network slicing, and a truncated sample is not worth
    tearing down a call over.
    """
    if len(pcm) % SAMPLE_WIDTH:
        pcm = pcm[: len(pcm) - (len(pcm) % SAMPLE_WIDTH)]
    return np.frombuffer(pcm, dtype="<i2")


def rms(pcm: bytes) -> float:
    """Root-mean-square amplitude, on the same 0..32767 scale as ``audioop.rms``.

    Kept on that scale so thresholds calibrated against the old gateway's logs
    remain meaningful. Computed in float64 because squaring int16 overflows.
    """
    samples = to_array(pcm)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def to_float32(pcm: bytes) -> np.ndarray:
    """Normalise to the float32 range [-1, 1) that ASR models expect."""
    return to_array(pcm).astype(np.float32) / _INT16_MAX


def from_float32(samples: np.ndarray) -> bytes:
    """Inverse of :func:`to_float32`, clipping rather than wrapping on overflow."""
    clipped: np.ndarray = np.clip(samples, -1.0, 1.0 - 1.0 / _INT16_MAX)
    scaled: np.ndarray = clipped * _INT16_MAX
    return scaled.astype("<i2").tobytes()


def swap_endian(pcm: bytes) -> bytes:
    """Reinterpret big-endian 16-bit PCM as little-endian.

    Needed only if a provider ever sends network byte order; kept because
    diagnosing it after the fact from garbled audio is miserable.
    """
    return to_array(pcm).byteswap().tobytes()


def to_mono(pcm: bytes, channels: int) -> bytes:
    """Downmix interleaved PCM to mono by averaging channels."""
    if channels == 1:
        return pcm
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    samples = to_array(pcm)
    usable = samples.size - (samples.size % channels)
    frames = samples[:usable].reshape(-1, channels)
    return frames.mean(axis=1).round().astype("<i2").tobytes()


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample by linear interpolation.

    This matches what ``audioop.ratecv`` did, and is adequate for the conversions
    actually performed here (Piper's 22.05 kHz output down to 16 kHz). It is not
    band-limited, so it will alias on aggressive downsampling; if a provider is
    ever added that needs a large ratio, ``scipy.signal.resample_poly`` is the
    upgrade and the reason to take the scipy dependency.
    """
    if src_rate == dst_rate:
        return pcm
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"rates must be positive, got {src_rate} -> {dst_rate}")
    samples = to_array(pcm)
    if samples.size == 0:
        return b""
    # Step the output grid by a fixed src/dst ratio rather than spreading a
    # fixed number of points across the input with linspace. Endpoint-inclusive
    # linspace implies a step of (n-1)/(out_len-1), which is not quite the
    # requested rate; over a few thousand samples the two grids drift apart, and
    # this function would then disagree with StreamingResampler on the same
    # audio. A constant step is what "resample to 16 kHz" actually means.
    ratio = src_rate / dst_rate
    out_len = int((samples.size - 1) / ratio) + 1
    if out_len <= 0:
        return b""
    positions = np.arange(out_len) * ratio
    resampled: np.ndarray = np.interp(
        positions, np.arange(samples.size), samples.astype(np.float64)
    )
    return resampled.round().astype("<i2").tobytes()


class StreamingResampler:
    """Resample a continuous PCM stream, chunk by chunk, without seams.

    Resampling each chunk independently produces an audible click at every
    boundary: the interpolator restarts at sample zero, so the fractional phase
    resets and the last input sample of one chunk is never interpolated against
    the first of the next. Over a streamed TTS clause arriving in 4 KB pieces
    that is a click every few milliseconds.

    This keeps the fractional read position and the unconsumed tail between
    calls, so the output is identical to resampling the whole stream at once.
    """

    __slots__ = ("_buffer", "_position", "_ratio")

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError(f"rates must be positive, got {src_rate} -> {dst_rate}")
        self._ratio = src_rate / dst_rate
        self._buffer = np.zeros(0, dtype=np.float64)
        self._position = 0.0

    @property
    def passthrough(self) -> bool:
        return self._ratio == 1.0

    def process(self, pcm: bytes) -> bytes:
        """Resample a chunk, holding back whatever the next chunk needs."""
        if self.passthrough:
            return pcm
        samples = to_array(pcm)
        if samples.size == 0:
            return b""

        self._buffer = np.concatenate([self._buffer, samples.astype(np.float64)])
        # Interpolation needs a sample on both sides, so the last input sample is
        # not consumable until more arrives.
        available = len(self._buffer) - 1 - self._position
        if available < 0:
            return b""

        count = int(available // self._ratio) + 1
        if count <= 0:
            return b""

        positions = self._position + np.arange(count) * self._ratio
        output: np.ndarray = np.interp(positions, np.arange(len(self._buffer)), self._buffer)

        consumed = int(positions[-1])
        self._buffer = self._buffer[consumed:]
        self._position = positions[-1] - consumed + self._ratio
        return output.round().astype("<i2").tobytes()

    def flush(self) -> bytes:
        """Emit the final samples once no more input is coming."""
        if self.passthrough or len(self._buffer) == 0:
            self._buffer = np.zeros(0, dtype=np.float64)
            return b""
        remaining = len(self._buffer) - self._position
        count = max(0, int(remaining // self._ratio))
        if count == 0:
            self._buffer = np.zeros(0, dtype=np.float64)
            return b""
        positions = self._position + np.arange(count) * self._ratio
        positions = np.clip(positions, 0, len(self._buffer) - 1)
        output: np.ndarray = np.interp(positions, np.arange(len(self._buffer)), self._buffer)
        self._buffer = np.zeros(0, dtype=np.float64)
        self._position = 0.0
        return output.round().astype("<i2").tobytes()


def silence(duration_ms: int) -> bytes:
    """Digital silence, for inter-sentence padding and playback tails."""
    if duration_ms <= 0:
        return b""
    return b"\x00" * (SAMPLE_RATE * duration_ms // 1000 * SAMPLE_WIDTH)


def decode_wav(data: bytes) -> bytes:
    """Decode a WAV container to canonical 16 kHz mono 16-bit PCM.

    TTS providers return whatever their model produces; the playback path only
    speaks one format, so normalisation happens once, here.
    """
    with wave.open(io.BytesIO(data), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        pcm = reader.readframes(reader.getnframes())

    if width != SAMPLE_WIDTH:
        pcm = _requantise(pcm, width)
    pcm = to_mono(pcm, channels)
    return resample(pcm, rate, SAMPLE_RATE)


def encode_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(CHANNELS)
        writer.setsampwidth(SAMPLE_WIDTH)
        writer.setframerate(rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


def _requantise(pcm: bytes, width: int) -> bytes:
    """Convert 8- or 32-bit PCM to 16-bit."""
    if width == 1:
        # WAV 8-bit is unsigned, centred on 128.
        samples = np.frombuffer(pcm, dtype=np.uint8).astype(np.int16)
        return ((samples - 128) << 8).astype("<i2").tobytes()
    if width == 4:
        samples = np.frombuffer(pcm, dtype="<i4")
        return (samples >> 16).astype("<i2").tobytes()
    raise ValueError(f"unsupported sample width: {width} bytes")
