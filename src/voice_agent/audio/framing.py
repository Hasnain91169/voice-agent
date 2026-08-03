"""Rechunking inbound audio into fixed 20 ms frames.

Vonage delivers 10 ms slices, browsers deliver whatever their buffer size and the
network conspire to produce. Everything downstream — VAD, echo gating, barge-in
detection — assumes a steady 20 ms frame, because energy thresholds calibrated
against one frame duration are meaningless at another.

This module is pure: bytes in, frames out, no sockets and no clock dependency
beyond stamping arrival. That makes the VAD and barge-in logic testable by
feeding synthetic PCM rather than standing up a WebSocket.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from voice_agent.audio import wav
from voice_agent.config import BYTES_PER_FRAME, FRAME_MS


@dataclass(frozen=True, slots=True)
class Frame:
    """One 20 ms window of caller audio, with its energy precomputed.

    RMS is computed once at rechunk time because both the utterance VAD and the
    barge-in detector need it, and they run concurrently over the same stream.
    """

    pcm: bytes
    rms: float
    seq: int
    #: ``time.monotonic()`` when the frame was completed.
    received_at: float

    @property
    def duration_ms(self) -> int:
        return FRAME_MS


class Rechunker:
    """Accumulates arbitrary-sized inbound slices and emits exact 20 ms frames.

    A partial frame is held until the remaining bytes arrive rather than being
    zero-padded: padding would inject artificial silence into the energy signal
    and cause spurious end-of-utterance detections on a jittery connection.
    """

    __slots__ = ("_buffer", "_seq", "_swap_endian")

    def __init__(self, *, swap_endian: bool = False) -> None:
        self._buffer = bytearray()
        self._seq = 0
        self._swap_endian = swap_endian

    def push(self, data: bytes) -> list[Frame]:
        """Feed inbound bytes; return whatever complete frames that produced."""
        if not data:
            return []
        if self._swap_endian:
            data = wav.swap_endian(data)
        self._buffer.extend(data)

        now = time.monotonic()
        frames: list[Frame] = []
        while len(self._buffer) >= BYTES_PER_FRAME:
            pcm = bytes(self._buffer[:BYTES_PER_FRAME])
            del self._buffer[:BYTES_PER_FRAME]
            frames.append(Frame(pcm=pcm, rms=wav.rms(pcm), seq=self._seq, received_at=now))
            self._seq += 1
        return frames

    def reset(self) -> None:
        """Drop buffered audio, keeping the sequence counter.

        Used when draining the backlog that accumulated while the agent was
        speaking: those bytes are our own voice returning and must not be
        interpreted as caller speech.
        """
        self._buffer.clear()

    @property
    def pending_bytes(self) -> int:
        return len(self._buffer)

    @property
    def frames_emitted(self) -> int:
        return self._seq


def slice_frames(pcm: bytes) -> Iterator[bytes]:
    """Split a PCM buffer into whole 20 ms frames, discarding any remainder.

    Used on the playback side, where the remainder is genuinely negligible (the
    tail of a synthesised clause) and holding it would stall the queue.
    """
    for offset in range(0, len(pcm) - BYTES_PER_FRAME + 1, BYTES_PER_FRAME):
        yield pcm[offset : offset + BYTES_PER_FRAME]
