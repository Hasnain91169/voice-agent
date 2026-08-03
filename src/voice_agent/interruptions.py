"""Conservative semantic gating for speech detected during playback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class InterruptionDecision(StrEnum):
    """What the pipeline should do with a candidate spoken over playback."""

    CONTINUE = "continue"
    INTERRUPT = "interrupt"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class InterruptionAssessment:
    """A deterministic, explainable decision for one candidate utterance."""

    decision: InterruptionDecision
    reason: str


# These are deliberately whole-utterance matches. A phrase such as
# "okay, show me Marchwood" must remain an interruption even though it starts
# with an acknowledgement.
_BACKCHANNELS = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "okay",
        "ok",
        "right",
        "got it",
        "sure",
        "great",
        "fine",
        "lovely",
        "solid",
        "thanks",
        "thank you",
        "mm hmm",
        "mm-hmm",
        "uh huh",
        "uh-huh",
        "ja",
        "genau",
        "klar",
        "gut",
        "verstanden",
        "danke",
        "danke schoen",
        "danke schön",
    }
)

_FILLERS = frozenset(
    {
        "uh",
        "um",
        "erm",
        "er",
        "hmm",
        "hm",
        "mhm",
        "oh",
        "ah",
    }
)

_HARD_INTERRUPTS = re.compile(
    r"\b(?:stop|wait|hold on|hang on|pause|cancel|no|what|sorry|repeat|"
    r"stopp|warte|moment|abbrechen|nein|wiederholen)\b",
    re.IGNORECASE,
)

_QUESTION_START = re.compile(
    r"^(?:who|what|when|where|which|why|how|can|could|would|do|does|is|are|"
    r"will|show|tell|give|check|find|wer|was|wann|wo|welche|warum|wie|"
    r"kann|könntest|zeige|sag|gib|prüfe|finde)\b",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    """Fold punctuation and whitespace without changing the spoken words."""

    return re.sub(r"[^\wäöüß'-]+", " ", text.casefold(), flags=re.UNICODE).strip()


def assess(text: str, confidence: float, *, min_confidence: float = 0.45) -> InterruptionAssessment:
    """Classify an overlaid utterance with a false-positive-safe policy.

    The classifier intentionally has no ``maybe interrupt`` result. Anything
    unclear continues the existing answer, which protects long spoken answers
    from being cut off by noise or a listener's acknowledgement.
    """

    normalised = _normalise(text)
    if not normalised or confidence < min_confidence:
        return InterruptionAssessment(InterruptionDecision.IGNORE, "low_confidence_or_empty")

    if normalised in _BACKCHANNELS:
        return InterruptionAssessment(InterruptionDecision.CONTINUE, "backchannel")
    if normalised in _FILLERS:
        return InterruptionAssessment(InterruptionDecision.CONTINUE, "filler")
    if _HARD_INTERRUPTS.search(normalised):
        return InterruptionAssessment(
            InterruptionDecision.INTERRUPT, "explicit_control_or_correction"
        )
    if "?" in text or _QUESTION_START.search(normalised):
        return InterruptionAssessment(InterruptionDecision.INTERRUPT, "new_question_or_request")

    # A single content word is often an account name, correction, or answer
    # that the rep wants handled now. Short non-content backchannels were
    # removed above, so this remains conservative without needing another model
    # in the real-time cancellation path.
    return InterruptionAssessment(InterruptionDecision.INTERRUPT, "new_content")


__all__ = ["InterruptionAssessment", "InterruptionDecision", "assess"]
