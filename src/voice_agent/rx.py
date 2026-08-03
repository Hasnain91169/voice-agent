"""The inbound audio pump.

**One task owns the transport's receive side for the entire call.** Everything
that needs caller audio — noise-floor calibration, utterance capture, barge-in
detection — reads from the frame channel this pump fills, and never from the
socket.

That is the single most important structural decision in the codebase. The
implementation this replaces had the playback loop peek at the socket with a
0.1ms timeout to spot interruptions, which meant playback and reception were
competing for one reader; frames went to whichever happened to be waiting.
Barge-in shipped disabled as a result. With a single owner, listening and
interrupt-detection are just two consumers of the same stream, and the pump
keeps running while the agent speaks — which is what makes interruption
detectable at all.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from voice_agent.audio.framing import Frame, Rechunker
from voice_agent.transports.base import Transport

log = logging.getLogger(__name__)

#: Roughly four seconds of audio. Deep enough to absorb a slow consumer (a CPU
#: transcribe can stall the loop for over a second), shallow enough that a
#: wedged consumer cannot grow it without bound.
DEFAULT_CAPACITY = 200


class FrameChannel:
    """A bounded frame queue that drops the oldest audio when it overflows.

    Dropping rather than blocking is deliberate. The pump must never wait on a
    consumer: if it did, a slow transcribe would stop the socket being read,
    the OS buffer would fill, and the caller's speech would arrive late and out
    of step for the rest of the call. When a choice has to be made, the newest
    audio is the audio worth keeping — it is the only audio that can still
    reveal that the caller has started talking.
    """

    __slots__ = ("_closed", "_queue", "dropped")

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._queue: asyncio.Queue[Frame | None] = asyncio.Queue(maxsize=capacity)
        self._closed = False
        self.dropped = 0

    def put(self, frame: Frame) -> None:
        """Enqueue a frame without ever blocking the pump."""
        if self._closed:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - racy but harmless
                pass
        self._queue.put_nowait(frame)

    async def get(self) -> Frame | None:
        """Await the next frame, or ``None`` once the call has ended."""
        return await self._queue.get()

    async def frames(self) -> AsyncIterator[Frame]:
        """Iterate frames until the call ends."""
        while (frame := await self.get()) is not None:
            yield frame

    def drain(self) -> int:
        """Discard everything buffered; returns how many frames went.

        Used after the agent finishes speaking: whatever accumulated during
        playback is our own voice returning, or audio recorded while the caller
        could not be heard properly. Either way it must not be transcribed.
        """
        discarded = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return discarded
            if item is None:  # preserve the end-of-call sentinel
                self._queue.put_nowait(None)
                return discarded
            discarded += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with_room = not self._queue.full()
        if not with_room:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover
                pass
        self._queue.put_nowait(None)

    @property
    def closed(self) -> bool:
        return self._closed

    def qsize(self) -> int:
        return self._queue.qsize()


class RxPump:
    """Reads the transport, rechunks to 20ms frames, and fans out via a channel."""

    def __init__(
        self,
        transport: Transport,
        channel: FrameChannel,
        *,
        swap_endian: bool = False,
    ) -> None:
        self._transport = transport
        self._channel = channel
        self._rechunker = Rechunker(swap_endian=swap_endian)

    async def run(self) -> None:
        """Pump until the transport closes or the task is cancelled."""
        try:
            while True:
                data = await self._transport.recv()
                if data is None:
                    break
                for frame in self._rechunker.push(data):
                    self._channel.put(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("rx pump failed")
        finally:
            self._channel.close()
            if self._channel.dropped:
                # Sustained drops mean a consumer is too slow to keep up, which
                # degrades endpointing and barge-in — worth surfacing.
                log.warning(
                    "rx dropped %d frames (%.1fs of audio)",
                    self._channel.dropped,
                    self._channel.dropped * 0.02,
                )
