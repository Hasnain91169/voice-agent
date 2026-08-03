"""Tests for the live grounding trace.

This replaces a panel in the supplied dashboard mockup that showed "Correctness
96%", "Hallucination Risk: Low" and an overall score of 4.8 out of 5, all badged
live. None of those exist. Correctness here comes from the offline eval suite —
an LLM judge over scripted scenarios with known ground truth — and cannot be
computed for a conversation nobody scripted.

What is computed instead is narrower and true: every figure the agent said,
checked against the text its tools returned. The tests that matter are the ones
about *not* crying wolf, because the agent is explicitly instructed to round for
speech, and a trace that flags obedience as invention is a guardrail nobody will
keep looking at.
"""

from __future__ import annotations

import pytest

from voice_agent.agent.grounding import _fold_spoken_numbers, summarise, trace

TOOLS = [
    "brief_account -> Marchwood Timber & Board, independent account. They've spent "
    "45 thousand pounds with us over the last year. They normally order every 12 "
    "days, and the last one was the 18th of May, 76 days ago."
]


def untraced(spoken: str, tools: list[str] | None = None) -> list[str]:
    return [f.text for f in trace(spoken, tools if tools is not None else TOOLS) if not f.traced]


# ─────────────────────────────────────────────────── spoken numbers to digits


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("forty-five thousand", "45000"),
        ("eighty five days", "85"),
        ("seven thousand two hundred", "7200"),
        ("twelve", "12"),
        ("one million", "1000000"),
        ("fünfundachtzig", "fünfundachtzig"),  # unrecognised: left alone, not guessed
    ],
)
def test_number_words_fold_into_one_figure(spoken: str, expected: str) -> None:
    """Compounds fold rather than substituting word by word.

    Replacing "forty" and "five" independently produced the figures 40 and 5 —
    neither of which any tool returned — so correct speech was flagged as
    invention.
    """
    assert expected in _fold_spoken_numbers(spoken)


def test_a_number_keeps_the_space_after_it() -> None:
    """ "twelve days" folded to "12days", and the word-boundary scan then
    skipped it entirely. A missed figure is a silent gap, which is worse than a
    false one because nothing prompts anyone to look."""
    assert _fold_spoken_numbers("twelve days") == "12 days"


# ──────────────────────────────────────────────────────────── not crying wolf


def test_rounding_for_speech_is_not_invention() -> None:
    """The system prompt tells the agent to say "about forty-five thousand"."""
    assert untraced("They've spent about forty-five thousand pounds") == []


def test_repeating_a_figure_verbatim_is_traced() -> None:
    assert untraced("They last ordered 76 days ago, on the 18th of May") == []


def test_a_turn_with_no_figures_traces_nothing() -> None:
    """Silence and refusals are not suspicious."""
    assert trace("I can't see anything on that account.", TOOLS) == []
    assert trace("", TOOLS) == []


# ────────────────────────────────────────────────────────── catching the real thing


def test_a_figure_no_tool_returned_is_flagged() -> None:
    said = "They've spent 45 thousand pounds, and ordered twice in June worth 9,300 pounds."
    # Reported in its normalised form: separators are stripped on both sides so
    # that "9,300" and "9.300" compare equal.
    assert any("9300" in figure for figure in untraced(said))


def test_a_figure_spoken_with_no_tools_called_at_all_is_flagged() -> None:
    """The clearest case: the agent produced a number from nowhere."""
    assert untraced("They ordered 14 times last quarter.", []) == ["14"]


def test_the_summary_names_the_untraced_figures() -> None:
    """A count is a score; a name is a prompt to go and look.

    The panel this replaces showed a score. The distinction is the whole point.
    """
    said = "They've spent 45 thousand pounds and ordered 14 times."
    result = summarise(trace(said, TOOLS))
    assert result["spoken"] == 2
    assert result["traced"] == 1
    assert result["untraced"] == ["14"]
