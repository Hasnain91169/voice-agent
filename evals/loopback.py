"""An in-process transport that plays both sides of a call.

Full-audio evaluation. The caller's turns are synthesised to PCM and fed in as
20ms frames; the agent's audio is captured on the way out. Because it satisfies
the same :class:`~voice_agent.transports.base.Transport` protocol as the browser
and telephony transports, the real pipeline runs unchanged — the VAD, the echo
gate, the endpointer and barge-in are all exercised for real rather than mocked.

That is the difference between an eval that tests the agent and one that tests
the *voice* agent. A text-level harness would never catch an endpointer that
cuts people off, or a barge-in threshold that fires on the agent's own voice.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from voice_agent.audio import wav
from voice_agent.config import BYTES_PER_FRAME, SAMPLE_RATE

log = logging.getLogger(__name__)

#: Amplitude of the dither used for "silence". Digital zero is not what any
#: real line carries, and calibrating against it produces a noise floor no
#: microphone ever sends.
_ROOM_TONE = 4


class LoopbackTransport:
    """Feeds scripted caller audio in, captures agent audio out."""

    name = "loopback"

    def __init__(self, *, echo_cancelled: bool = True) -> None:
        self._inbound: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._echo_cancelled = echo_cancelled
        self.spoken: bytearray = bytearray()
        #: Monotonic time of the most recent outbound frame, so a caller can
        #: tell whether the agent is still talking.
        self.last_spoke_at: float = 0.0
        #: Monotonic time of the *first* outbound frame since the clock was
        #: reset. Separate from ``last_spoke_at`` because the two answer
        #: different questions and conflating them silently mismeasures the
        #: headline latency figure: time-to-first-audio is what a caller
        #: experiences as responsiveness, while the last frame is simply when
        #: the answer finished.
        self.first_spoke_at: float = 0.0
        self.closed = False

    @property
    def echo_cancelled(self) -> bool:
        return self._echo_cancelled

    # ------------------------------------------------------------- transport

    async def recv(self) -> bytes | None:
        return await self._inbound.get()

    async def send(self, pcm: bytes) -> None:
        self.spoken += pcm
        now = asyncio.get_running_loop().time()
        if self.first_spoke_at == 0.0:
            self.first_spoke_at = now
        self.last_spoke_at = now

    async def close(self) -> None:
        self.closed = True
        await self._inbound.put(None)

    # ----------------------------------------------------------- caller side

    async def say(self, pcm: bytes, *, realtime: bool = True) -> None:
        """Feed caller audio in 20ms frames.

        Paced in real time by default. Sending faster would let a whole
        utterance arrive before the VAD has seen the start of it, which is not
        how a call behaves and would mask endpointing problems.
        """
        for offset in range(0, len(pcm) - BYTES_PER_FRAME + 1, BYTES_PER_FRAME):
            await self._inbound.put(pcm[offset : offset + BYTES_PER_FRAME])
            if realtime:
                await asyncio.sleep(0.02)

    async def hush(self, seconds: float, *, realtime: bool = True) -> None:
        """Feed room tone, which is what ends an utterance."""
        rng = np.random.default_rng(1234)
        for _ in range(int(seconds / 0.02)):
            noise = (rng.standard_normal(320) * _ROOM_TONE).astype("<i2").tobytes()
            await self._inbound.put(noise)
            if realtime:
                await asyncio.sleep(0.02)

    async def wait_until_quiet(self, *, idle: float = 0.35, give_up: float = 45.0) -> float:
        """Wait for the agent to stop speaking; return seconds of audio it sent.

        Polled rather than event-driven on purpose: the condition is the
        *absence* of frames for a period, which no event can signal — nothing
        fires when audio stops arriving.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + give_up
        # Wait for it to start, so a slow first token is not read as silence.
        while self.last_spoke_at == 0.0 and loop.time() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.02)
        while loop.time() < deadline:
            if loop.time() - self.last_spoke_at > idle:
                break
            await asyncio.sleep(0.02)
        return len(self.spoken) / (SAMPLE_RATE * 2)

    @property
    def speaking(self) -> bool:
        loop = asyncio.get_running_loop()
        return (loop.time() - self.last_spoke_at) < 0.25

    def take_spoken(self) -> bytes:
        """Consume everything captured since the last call."""
        audio = bytes(self.spoken)
        self.spoken = bytearray()
        return audio

    def reset_speech_clock(self) -> None:
        self.last_spoke_at = 0.0
        self.first_spoke_at = 0.0


def to_frames(pcm: bytes) -> int:
    return len(pcm) // BYTES_PER_FRAME


def duration_s(pcm: bytes) -> float:
    return len(pcm) / (SAMPLE_RATE * 2)


def rms(pcm: bytes) -> float:
    return wav.rms(pcm)
