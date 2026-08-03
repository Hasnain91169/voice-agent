"""Tests for the conservative semantic interruption gate."""

from __future__ import annotations

import pytest

from voice_agent.interruptions import InterruptionDecision, assess


@pytest.mark.parametrize(
    "text",
    [
        "yes",
        "okay",
        "right",
        "got it",
        "lovely",
        "solid",
        "thanks",
        "ja",
        "genau",
        "verstanden",
        "danke",
    ],
)
def test_whole_utterance_acknowledgements_keep_the_agent_speaking(text: str) -> None:
    result = assess(text, 0.9)
    assert result.decision is InterruptionDecision.CONTINUE
    assert result.reason == "backchannel"


@pytest.mark.parametrize(
    "text",
    [
        "okay, show me Marchwood instead",
        "yes, which accounts are slipping",
        "what did you say",
        "stop there",
        "warte, zeig mir Marchwood",
        "Marchwood",
    ],
)
def test_new_requests_and_corrections_interrupt(text: str) -> None:
    assert assess(text, 0.9).decision is InterruptionDecision.INTERRUPT


@pytest.mark.parametrize("text", ["", "uh", "hmm"])
def test_noise_and_fillers_do_not_cut_the_answer(text: str) -> None:
    result = assess(text, 0.1 if not text else 0.9)
    assert result.decision in {InterruptionDecision.IGNORE, InterruptionDecision.CONTINUE}


def test_uncertain_content_defaults_to_ignore() -> None:
    result = assess("Marchwood", 0.2)
    assert result.decision is InterruptionDecision.IGNORE

