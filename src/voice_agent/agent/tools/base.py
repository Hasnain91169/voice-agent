"""Tool definitions.

Tools are declared provider-neutrally and converted by each LLM adapter, so a
tool is written once regardless of which model is behind it.

Two constraints shape every result string here. It is **read aloud**, so a
result is a sentence rather than a JSON blob — the model would otherwise
paraphrase a table, badly, and slowly. And it goes back through the model,
so every token in a result is paid for twice: once to read it, once in the
latency of generating the reply. Results are therefore short by design.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from voice_agent.agent import locale


class ToolError(Exception):
    """A tool failed in a way the agent should hear about and work around."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One callable capability offered to the model."""

    name: str
    description: str
    #: JSON Schema for the arguments.
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[str]]

    def as_dict(self) -> dict[str, Any]:
        """The neutral shape the LLM adapters convert from."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


#: German month names. ``strftime('%B')`` is locale-dependent and would return
#: whatever the host machine happens to be set to — English on this one, and
#: something else on a server in Frankfurt. A table is deterministic.
_MONTHS_DE: Final = (
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
)


def money(amount: float) -> str:
    """Format a figure the way it should be spoken, not written.

    Every magnitude is rounded here, and that is the point. The system prompt
    tells the agent to say "about twelve thousand pounds" rather than read out a
    ledger figure, so whatever precision this function keeps, the model strips —
    and it strips it by doing arithmetic, which it sometimes gets wrong. Handing
    back "7,160.00 pounds" is what produced a spoken "seventy-nine hundred": the
    decimals are noise the model has to remove, and it mis-rounded on the way.

    Rounding here leaves nothing to convert. It is the same reasoning as
    :func:`ago` giving both the date and the age — the tool does the arithmetic
    precisely once, in code, where it can be tested.

    German uses a full stop as the thousands separator, so "7,200" and "7.200"
    are the same number written for different readers — and a synthesiser reads
    the wrong one as a decimal. The currency stays pounds: the accounts are a UK
    merchant's, and converting them would invent an exchange rate.
    """
    german = locale.current() == "de"
    if amount >= 1_000_000:
        millions = f"{amount / 1_000_000:.1f}"
        if german:
            return f"{millions.replace('.', ',')} Millionen Pfund"
        return f"{millions} million pounds"
    if amount >= 10_000:
        thousands = f"{amount / 1000:.0f}"
        return f"{thousands} tausend Pfund" if german else f"{thousands} thousand pounds"
    if amount >= 1_000:
        # Nearest hundred: enough for a rep to act on, too coarse to invite a
        # restatement that drifts.
        rounded = f"{round(amount / 100) * 100:,.0f}"
        return f"{rounded.replace(',', '.')} Pfund" if german else f"{rounded} pounds"
    whole = f"{amount:,.0f}"
    return f"{whole} Pfund" if german else f"{whole} pounds"


def spoken_date(value: str | datetime) -> str:
    """A past date the way someone says it: "the 6th of May", "6. Mai".

    Note ``%-d`` is not portable — it fails outright on Windows — and ``%d``
    gives "06", which a synthesiser reads as "oh six". The day is formatted by
    hand for both reasons.

    German writes an ordinal as a bare number and a full stop, "6. Mai", and
    reads it "sechsten Mai". The month name comes from a table rather than
    ``strftime`` so the output does not depend on the host's locale.
    """
    moment = datetime.fromisoformat(value) if isinstance(value, str) else value
    day = moment.day
    if locale.current() == "de":
        return f"{day}. {_MONTHS_DE[moment.month - 1]}"
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"the {day}{suffix} of {moment.strftime('%B')}"


def ago(days: int, when: str | datetime) -> str:
    """Both the date and the age, because the model cannot derive one from the
    other.

    Given only "77 days", a model converts it to an absolute date using its own
    sense of today — which is its training cutoff, not this dataset's frozen
    snapshot — and states it confidently. Given only a date, it has to do the
    subtraction. Supplying both removes the arithmetic entirely, and the eval
    judge stops catching invented dates like "early June".
    """
    if locale.current() == "de":
        return f"{spoken_date(when)}, vor {days} Tagen"
    return f"{spoken_date(when)}, {days} days ago"
