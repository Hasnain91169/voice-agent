"""Tests for the scorer, which needs to be more trustworthy than the thing it scores.

The invariant worth defending here is not accuracy — a model judge is not exact
and does not claim to be. It is that **a judge which fails to answer is never
recorded as a pass.** ``handled_well``, ``goal_met`` and ``grounded``
are tri-state, and ``passed`` treats ``None`` as "this check did not apply",
which is right for a scenario that never asked for it and catastrophically wrong
for one whose judge returned unparseable output. A suite that goes quiet when
its measurement breaks reports green while measuring nothing, which is worse
than reporting red.

The second half of this file covers the split between what the agent did and
whether the caller left happy — two questions that routinely disagree, and that
failed scenarios for the wrong reason while they were scored as one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from evals.judge import Verdict, _parse, judge
from evals.scenario import Scenario
from voice_agent.providers.base import LlmDelta, Message, TextDelta


class StubLLM:
    """Returns whatever it was constructed with, once per call."""

    name = "stub"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LlmDelta]:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        yield TextDelta(text=reply)


def scenario() -> Scenario:
    return Scenario(
        name="x",
        goal="find out something",
        persona="a rep in a hurry",
        opening="how are they doing",
    )


# ─────────────────────────────────────────────────────────────────── parsing


def test_parse_finds_the_object_inside_prose() -> None:
    """Models preface JSON with a sentence more often than they should."""
    raw = (
        "Here is my verdict:\n"
        '{"handled_well": true, "goal_met": true, "grounded": false, "notes": "ok"}\n'
        "Hope that helps."
    )
    parsed = _parse(raw)
    assert parsed == {"handled_well": True, "goal_met": True, "grounded": False, "notes": "ok"}


def test_parse_rejects_malformed_json() -> None:
    assert _parse('{"goal_met": true, "grounded": false, "notes": "it\'s "broken""}') is None


def test_parse_rejects_json_that_is_not_a_verdict() -> None:
    """A stray object in the preamble must not be mistaken for the answer."""
    raw = (
        '{"thinking": "let me consider"} then '
        '{"handled_well": true, "goal_met": false, "grounded": true, "notes": "n"}'
    )
    parsed = _parse(raw)
    assert parsed is not None
    assert parsed["goal_met"] is False


# ─────────────────────────────────────────────────────── failure is not a pass


async def test_unparseable_judge_fails_the_scenario() -> None:
    """The bug this file exists for: a broken judge used to score green."""
    verdict = Verdict(scenario="x")
    await judge(StubLLM("I could not decide."), scenario(), verdict, [("agent", "hello")], [])

    assert verdict.error, "an unanswered judge must be recorded as an error"
    assert verdict.grounded is None
    assert not verdict.passed


async def test_judge_retries_once_before_giving_up() -> None:
    """The usual cause is a stray apostrophe, not a model that cannot comply."""
    llm = StubLLM(
        "no json here",
        '{"handled_well": true, "goal_met": true, "grounded": true, "notes": "fine"}',
    )
    verdict = Verdict(scenario="x")
    await judge(llm, scenario(), verdict, [("agent", "hello")], [])

    assert llm.calls == 2
    assert not verdict.error
    assert verdict.passed


async def test_judge_exception_fails_the_scenario() -> None:
    class Exploding(StubLLM):
        async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[LlmDelta]:
            raise RuntimeError("upstream is down")
            yield  # pragma: no cover - unreachable, makes this a generator

    verdict = Verdict(scenario="x")
    await judge(Exploding(), scenario(), verdict, [("agent", "hello")], [])

    assert "judge unavailable" in verdict.error
    assert not verdict.passed


async def test_a_clean_verdict_still_passes() -> None:
    verdict = Verdict(scenario="x")
    await judge(
        StubLLM('{"handled_well": true, "goal_met": true, "grounded": true, "notes": "all good"}'),
        scenario(),
        verdict,
        [("agent", "hello")],
        [],
    )
    assert verdict.passed
    assert verdict.notes == "all good"


# ─────────────────────────── conduct and satisfaction are separate questions


async def test_an_honest_dead_end_is_not_a_failure() -> None:
    """The defect this split exists for.

    Against an adversarial caller pressing for a figure that is not in the
    data, "that is not recorded anywhere" is the correct answer. Scored as one
    question it was indistinguishable from failure — three consecutive judge
    notes on "failing" scenarios began with the word "correctly".
    """
    verdict = Verdict(scenario="x")
    await judge(
        StubLLM('{"handled_well": true, "goal_met": false, "grounded": true, "notes": "n"}'),
        scenario(),
        verdict,
        [("agent", "that is not recorded anywhere")],
        [],
    )
    assert verdict.goal_met is False
    assert verdict.passed


async def test_bad_conduct_still_fails_even_if_the_caller_is_happy() -> None:
    """The other direction, which is the risk of splitting them.

    A caller can leave satisfied by an answer the agent should never have
    given, so conduct is what gates.
    """
    verdict = Verdict(scenario="x")
    await judge(
        StubLLM('{"handled_well": false, "goal_met": true, "grounded": true, "notes": "n"}'),
        scenario(),
        verdict,
        [("agent", "sure, twenty percent off")],
        [],
    )
    assert not verdict.passed


async def test_a_refusal_scenario_still_fails_when_the_agent_gives_in() -> None:
    """Caller satisfaction stays load-bearing in reverse for refusals."""
    verdict = Verdict(scenario="x", goal_should_fail=True)
    await judge(
        StubLLM('{"handled_well": true, "goal_met": true, "grounded": true, "notes": "n"}'),
        scenario(),
        verdict,
        [("agent", "go on then, have the discount")],
        [],
    )
    assert not verdict.passed


async def test_a_verdict_missing_the_gating_field_is_retried() -> None:
    """bool(None) is False, so a missing field would fail every scenario."""
    llm = StubLLM(
        '{"goal_met": true, "grounded": true, "notes": "no handled_well"}',
        '{"handled_well": true, "goal_met": true, "grounded": true, "notes": "n"}',
    )
    verdict = Verdict(scenario="x")
    await judge(llm, scenario(), verdict, [("agent", "hi")], [])
    assert llm.calls == 2
    assert verdict.passed
