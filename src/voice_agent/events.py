"""What the pipeline is doing, as it does it.

Everything worth showing about a call already exists inside the pipeline —
per-stage latencies, which tool ran, the moment barge-in cancelled a turn, the
gap between what was generated and what was heard. None of it ever left the
process. It went to a log file, which is the wrong shape for a person watching a
call happen.

This is the channel out. Three properties keep it from leaking into the parts of
the system that should not know about it:

**Optional.** The default sink discards. The loopback transport used by the eval
harness and the telephony transport neither emit nor consume events, and nothing
in the turn loop branches on whether anyone is listening.

**Fire and forget.** Emitting never blocks the pipeline and never raises into it.
A browser that has stopped reading must not be able to stall a call — the audio
path has a 20ms deadline and a dashboard does not.

**Descriptive, not authoritative.** Events report what happened. Nothing reads
them back; ``TurnMetrics`` and the conversation store remain the record. A
subscriber that misses an event sees a gap in a display, not a wrong system.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

log = logging.getLogger(__name__)


class EventSink(Protocol):
    """Somewhere for pipeline events to go."""

    def __call__(self, kind: str, /, **fields: Any) -> None: ...


def discard(kind: str, /, **fields: Any) -> None:
    """The default. Costs one function call per event and does nothing."""


class Emitter:
    """Wraps a sink so a broken subscriber cannot take a call down.

    The pipeline emits from inside the turn loop, where an exception would
    cancel the turn and a slow call would eat the latency budget. Anything the
    sink does wrong is logged once and swallowed.
    """

    __slots__ = ("_sink",)

    def __init__(self, sink: EventSink | None = None) -> None:
        self._sink: EventSink = sink or discard

    def __call__(self, kind: str, /, **fields: Any) -> None:
        try:
            self._sink(kind, **fields)
        except Exception:
            # Once, at debug: a subscriber failing every frame would otherwise
            # fill the log with the same traceback at 50 lines a second.
            log.debug("event sink failed on %r", kind, exc_info=True)


def buffered(limit: int = 500) -> tuple[Emitter, Callable[[], list[dict[str, Any]]]]:
    """An emitter that keeps the last ``limit`` events, for tests.

    Returns the emitter and a function that drains what it has seen.
    """
    seen: list[dict[str, Any]] = []

    def sink(kind: str, /, **fields: Any) -> None:
        seen.append({"type": kind, **fields})
        if len(seen) > limit:
            del seen[0]

    def drain() -> list[dict[str, Any]]:
        out = list(seen)
        seen.clear()
        return out

    return Emitter(sink), drain
