"""Transport protocol.

A transport is a bidirectional stream of 16 kHz mono 16-bit PCM. The pipeline
does not know whether it is talking to a browser over a WebSocket or to a
carrier over Vonage's media stream, which is what lets the same turn loop and
the same barge-in logic serve both.

``echo_cancelled`` is the one place the difference leaks through, and it must:
a browser negotiates real acoustic echo cancellation through ``getUserMedia``,
so the agent's own voice never returns on the inbound stream. Telephony has no
such guarantee, so the echo gate has to suppress inbound audio after playback
and be suspicious of what follows. Applying the telephony guard to a browser
session would cost latency for nothing; omitting it on telephony would make the
agent interrupt itself.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    """A duplex PCM stream for one call."""

    name: str

    @property
    def echo_cancelled(self) -> bool:
        """Whether the far end removes our own audio from what it sends back."""
        ...

    async def recv(self) -> bytes | None:
        """Await the next inbound audio, or ``None`` once the call has ended.

        Only ever called by :class:`voice_agent.rx.RxPump`. A second consumer
        would steal frames from the pump, which is precisely the design fault
        that made the previous implementation's barge-in unusable.
        """
        ...

    async def send(self, pcm: bytes) -> None:
        """Send one frame of outbound audio."""
        ...

    async def close(self) -> None: ...
