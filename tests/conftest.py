"""Shared fixtures and PCM builders.

Tests construct audio arithmetically rather than loading recordings, so they run
anywhere, stay fast, and describe their intent in the call site: a frame with an
RMS of 900 is self-evidently loud against a floor of 20.
"""

from __future__ import annotations

import numpy as np

from voice_agent.audio.framing import Frame
from voice_agent.config import SAMPLES_PER_FRAME


def square_pcm(amplitude: int, *, frames: int = 1) -> bytes:
    """A square wave of the given amplitude, whose RMS equals that amplitude."""
    count = SAMPLES_PER_FRAME * frames
    samples = np.empty(count, dtype="<i2")
    samples[0::2] = amplitude
    samples[1::2] = -amplitude
    return samples.tobytes()


def make_frame(amplitude: int, seq: int = 0, *, received_at: float = 0.0) -> Frame:
    """A frame whose energy is exactly ``amplitude``."""
    return Frame(
        pcm=square_pcm(amplitude),
        rms=float(amplitude),
        seq=seq,
        received_at=received_at,
    )


def make_frames(amplitudes: list[int], *, start_seq: int = 0) -> list[Frame]:
    return [make_frame(a, start_seq + i) for i, a in enumerate(amplitudes)]
