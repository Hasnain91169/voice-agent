"""Browser transport over a WebSocket.

Exists so the repository can be cloned and talked to in five minutes without a
phone number or a carrier account. It is also the honest place to test barge-in
interactively, because a browser negotiates real acoustic echo cancellation —
which telephony does not — so interruptions can be tuned without the agent's own
voice confusing the measurement.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

log = logging.getLogger(__name__)


class BrowserTransport:
    """Duplex 16 kHz PCM over a WebSocket."""

    name = "browser"

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        #: In-flight event sends, held so they are not garbage collected
        #: mid-flight and so close() can wait for them.
        self._pending: set[asyncio.Task[None]] = set()

    @property
    def echo_cancelled(self) -> bool:
        """The page requests ``echoCancellation`` when opening the microphone.

        This lets the session skip the post-playback guard window that a phone
        line needs, which is pure latency here.
        """
        return True

    async def recv(self) -> bytes | None:
        """Next inbound audio frame, or ``None`` when the socket closes."""
        try:
            message = await self._ws.receive()
        except (WebSocketDisconnect, RuntimeError):
            return None

        if message["type"] == "websocket.disconnect":
            return None

        data: bytes | None = message.get("bytes")
        if data is not None:
            return data

        # Text frames are control messages, not audio. Returning empty bytes
        # keeps the pump running without feeding the rechunker nonsense.
        text = message.get("text")
        if text:
            log.debug("control: %s", text[:200])
        return b""

    async def send(self, pcm: bytes) -> None:
        if self._ws.client_state is not WebSocketState.CONNECTED:
            raise WebSocketDisconnect(code=1000)
        await self._ws.send_bytes(pcm)

    def send_event(self, kind: str, /, **fields: Any) -> None:
        """Push a pipeline event to the page over the same socket.

        Binary frames are audio and text frames are events, which the browser
        separates without a wrapper or a second connection.

        Scheduled rather than awaited, because this is called from inside the
        turn loop where the audio path has a 20ms deadline. A page that has
        stopped reading must be able to fall behind without stalling a call —
        the send is fire and forget, and a failed one is a gap in a display.
        """
        if self._ws.client_state is not WebSocketState.CONNECTED:
            return
        with contextlib.suppress(RuntimeError):  # no running loop
            task = asyncio.get_running_loop().create_task(self._send_json({"type": kind, **fields}))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    async def _send_json(self, payload: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            await self._ws.send_json(payload)

    async def close(self) -> None:
        if self._pending:
            await asyncio.gather(*tuple(self._pending), return_exceptions=True)
        if self._ws.client_state is WebSocketState.CONNECTED:
            try:
                await self._ws.close()
            except RuntimeError:  # pragma: no cover - already closing
                pass
