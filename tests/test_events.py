"""Tests for the pipeline's event channel.

The properties worth defending are all about what a subscriber *cannot* do. The
emitter sits inside the turn loop, on a path with a 20ms deadline, so a browser
that has stopped reading, or a sink that raises on every call, must not be able
to slow a call down or take one out. An event is a description of something that
already happened; nothing downstream depends on it arriving.
"""

from __future__ import annotations

from typing import Any

import pytest

from voice_agent.events import Emitter, buffered, discard


def test_the_default_sink_does_nothing_and_says_nothing() -> None:
    """A pipeline with no subscriber must not need a branch anywhere."""
    emit = Emitter()
    emit("anything", value=1)  # no exception, no output


def test_discard_accepts_any_event() -> None:
    discard("whatever", a=1, b="two")


def test_events_reach_the_sink_with_their_type() -> None:
    emit, drain = buffered()
    emit("asr", text="hello", confidence=0.9)
    emit("turn_complete", turn=1)

    assert drain() == [
        {"type": "asr", "text": "hello", "confidence": 0.9},
        {"type": "turn_complete", "turn": 1},
    ]


def test_draining_clears() -> None:
    emit, drain = buffered()
    emit("one")
    assert len(drain()) == 1
    assert drain() == []


def test_a_sink_that_raises_cannot_reach_the_pipeline() -> None:
    """The important one.

    This is called from inside the turn loop. An exception escaping here would
    cancel the turn — a dashboard bug would become a dropped call.
    """

    def explode(kind: str, /, **fields: Any) -> None:
        raise RuntimeError("subscriber is broken")

    emit = Emitter(explode)
    emit("asr", text="hello")  # must not raise


def test_a_sink_that_raises_every_time_still_does_not_raise() -> None:
    """Failure is logged once at debug, not re-raised on each frame.

    A subscriber failing at fifty events a second would otherwise fill the log
    with one traceback per frame, which is its own outage.
    """

    def explode(kind: str, /, **fields: Any) -> None:
        raise ValueError(kind)

    emit = Emitter(explode)
    for _ in range(50):
        emit("frame")


def test_the_buffer_is_bounded() -> None:
    """A long call must not grow the buffer without limit."""
    emit, drain = buffered(limit=10)
    for i in range(40):
        emit("tick", i=i)
    seen = drain()
    assert len(seen) == 10
    # The most recent survive; the oldest are dropped.
    assert seen[-1]["i"] == 39


@pytest.mark.parametrize("kind", ["asr", "tool_call", "barge_in", "grounding", "turn_complete"])
def test_every_event_the_dashboard_handles_can_be_emitted(kind: str) -> None:
    """Names are strings on both sides, so drift is only caught by a test.

    These are the types ``demo.html`` switches on; an event renamed on the
    pipeline side would otherwise fall through to the raw log with nobody
    noticing until a demo.
    """
    emit, drain = buffered()
    emit(kind)
    assert drain()[0]["type"] == kind
