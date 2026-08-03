"""Tests for the frame channel and the inbound pump."""

from __future__ import annotations

import asyncio

from tests.conftest import make_frame, square_pcm
from voice_agent.config import BYTES_PER_FRAME
from voice_agent.rx import FrameChannel, RxPump


class FakeTransport:
    """A transport that replays a scripted sequence of inbound buffers."""

    name = "fake"
    echo_cancelled = True

    def __init__(self, chunks: list[bytes | None]) -> None:
        self._chunks = list(chunks)
        self.sent: list[bytes] = []
        self.closed = False

    async def recv(self) -> bytes | None:
        if not self._chunks:
            return None
        return self._chunks.pop(0)

    async def send(self, pcm: bytes) -> None:
        self.sent.append(pcm)

    async def close(self) -> None:
        self.closed = True


class TestFrameChannel:
    async def test_round_trips_frames_in_order(self) -> None:
        channel = FrameChannel()
        for i in range(3):
            channel.put(make_frame(100, i))
        assert [(await channel.get()).seq for _ in range(3)] == [0, 1, 2]  # type: ignore[union-attr]

    async def test_close_yields_a_sentinel(self) -> None:
        channel = FrameChannel()
        channel.close()
        assert await channel.get() is None

    async def test_overflow_drops_oldest_rather_than_blocking(self) -> None:
        # The pump must never wait on a consumer: a slow transcribe would
        # otherwise stop the socket being read and desynchronise the call.
        channel = FrameChannel(capacity=4)
        for i in range(10):
            channel.put(make_frame(100, i))
        assert channel.dropped == 6
        first = await channel.get()
        assert first is not None
        # The newest audio survives, because only recent audio can still reveal
        # that the caller has started speaking.
        assert first.seq == 6

    async def test_drain_discards_backlog(self) -> None:
        channel = FrameChannel()
        for i in range(5):
            channel.put(make_frame(100, i))
        assert channel.drain() == 5
        assert channel.qsize() == 0

    async def test_drain_preserves_the_end_of_call_sentinel(self) -> None:
        # Losing the sentinel would hang the call loop forever.
        channel = FrameChannel()
        channel.put(make_frame(100, 0))
        channel.close()
        channel.drain()
        assert await channel.get() is None

    async def test_put_after_close_is_ignored(self) -> None:
        channel = FrameChannel()
        channel.close()
        channel.put(make_frame(100, 0))
        assert await channel.get() is None

    async def test_frames_iterator_stops_at_close(self) -> None:
        channel = FrameChannel()
        channel.put(make_frame(100, 0))
        channel.put(make_frame(100, 1))
        channel.close()
        seen = [frame.seq async for frame in channel.frames()]
        assert seen == [0, 1]


class TestRxPump:
    async def test_rechunks_inbound_audio_into_frames(self) -> None:
        # Vonage sends 10ms slices; the pump must present 20ms frames.
        half = BYTES_PER_FRAME // 2
        transport = FakeTransport([b"\x01\x02" * (half // 2)] * 4)
        channel = FrameChannel()
        await RxPump(transport, channel).run()

        frames = [f.seq async for f in channel.frames()]
        assert frames == [0, 1]

    async def test_closes_the_channel_when_the_transport_ends(self) -> None:
        channel = FrameChannel()
        await RxPump(FakeTransport([]), channel).run()
        assert await channel.get() is None

    async def test_computes_energy_for_downstream_consumers(self) -> None:
        transport = FakeTransport([square_pcm(2_500)])
        channel = FrameChannel()
        await RxPump(transport, channel).run()
        frame = await channel.get()
        assert frame is not None
        assert frame.rms == 2_500.0

    async def test_empty_buffers_are_harmless(self) -> None:
        # Control frames arrive as empty byte strings from the browser transport.
        transport = FakeTransport([b"", b"", square_pcm(500)])
        channel = FrameChannel()
        await RxPump(transport, channel).run()
        assert (await channel.get()) is not None

    async def test_cancellation_still_closes_the_channel(self) -> None:
        class Blocking(FakeTransport):
            async def recv(self) -> bytes | None:
                await asyncio.sleep(60)
                return None

        channel = FrameChannel()
        task = asyncio.create_task(RxPump(Blocking([]), channel).run())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert await channel.get() is None
