"""Which figures the agent spoke, and whether the tools gave it them.

The dashboard mockup this replaces showed "Correctness 96%" and "Hallucination
Risk: Low" with a live badge. Neither exists. Correctness in this repo comes from
the offline eval suite — an LLM judge, scripted scenarios, known ground truth —
and cannot be computed for an ad-hoc conversation. Showing an invented number
beside real ones is the fastest way to make a reviewer distrust all of them.

What *can* be computed live, exactly and cheaply, is narrower and more useful:
every number and date the agent said, checked against the text its tools returned
that turn. Not "is this true" — "did this come from somewhere".

It is deliberately not a score. It is a list of figures, each traced or not, and
an untraced one is a prompt to look rather than a verdict. The agent is allowed
to say "about seven thousand" for `7,200 pounds`, so matching is on the digits a
figure is built from rather than on the string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Numbers, money and dates as they appear in spoken output. Deliberately loose:
#: a missed figure is a gap in a display, a false match is a claim that
#: something was verified when it was not.
#: Months, spelled out. A date is matched against these rather than against
#: "a number followed by a capitalised word", because ``re.IGNORECASE`` defeats
#: the capitalisation and "14 times" then matches as a date.
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|januar|februar|märz|mai|juni|juli|oktober|dezember"
)

_FIGURE = re.compile(
    rf"""
    (?P<money>\d[\d.,]*\s*(?:pounds|pfund|euro))
  | (?P<date>\d{{1,2}}(?:st|nd|rd|th)?\.?\s+(?:of\s+)?(?:{_MONTHS})\b)
  | (?P<percent>\d[\d.,]*\s*(?:percent|prozent|%))
  | (?P<plain>\b\d[\d.,]*\b)
    """,
    re.VERBOSE | re.IGNORECASE,
)

#: Number words, in both languages. Compounds are folded rather than substituted
#: word by word: replacing "forty" and "five" independently turns "forty-five
#: thousand pounds" into the figures 40 and 5, neither of which any tool
#: returned — a guardrail flagging correct speech as invention, which is worse
#: than no guardrail.
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
    "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6,
    "sieben": 7, "acht": 8, "neun": 9, "zehn": 10, "elf": 11, "zwölf": 12,
}  # fmt: skip

_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
    "zwanzig": 20, "dreißig": 30, "vierzig": 40, "fünfzig": 50,
    "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90,
}  # fmt: skip

_SCALES = {
    "hundred": 100, "thousand": 1_000, "million": 1_000_000,
    "hundert": 100, "tausend": 1_000, "millionen": 1_000_000,
}  # fmt: skip

#: Characters that may sit inside a run of number words without ending it.
#: "forty-five" is one number; splitting on the hyphen produced 40 and 5.
_JOINERS = frozenset(" -\t\n")


def _fold_spoken_numbers(text: str) -> str:
    """Turn runs of number words into digits.

    Small and deliberately incomplete — it covers what this agent actually says,
    which is quantities of pounds and counts of days. It does not attempt
    ordinals, fractions, or "a couple of hundred", and anything it does not
    recognise is left alone rather than guessed at.
    """
    # Collapse separators sitting between digits first: tokenising splits
    # "9,300" into 9 and 300, and two figures neither tool returned appear where
    # one it did should have been. English uses a comma and German a full stop
    # for the same job, and since comparison is on digits either way, both are
    # simply removed — on the tool output and the speech alike.
    text = re.sub(r"(?<=\d)[.,](?=\d)", "", text)
    tokens = re.split(r"(\W+)", text)
    out: list[str] = []
    total = 0
    current = 0
    seen = False

    def flush() -> None:
        """Emit the folded number, keeping a space after it.

        The separator that ended the run was swallowed while the run was being
        collected, so "twelve days" folded to "12days" — and ``\\b`` does not
        match between a digit and a letter, so the figure disappeared from the
        scan entirely. A missed figure is a silent gap, which is worse than a
        false one because nothing prompts anyone to look.
        """
        nonlocal total, current, seen
        if seen:
            out.append(f"{total + current} ")
        total = current = 0
        seen = False

    for token in tokens:
        word = token.lower()
        if word.isdigit():
            # A digit can start a run too: the agent says "45 thousand pounds"
            # as readily as "forty-five thousand", and folding only the words
            # split that into the figures 45 and 1000.
            flush()
            current = int(word)
            seen = True
        elif word in _UNITS:
            current += _UNITS[word]
            seen = True
        elif word in _TENS:
            current += _TENS[word]
            seen = True
        elif word in _SCALES:
            scale = _SCALES[word]
            if scale >= 1000:
                total = (total + max(current, 1)) * scale
                current = 0
            else:
                current = max(current, 1) * scale
            seen = True
        elif token and set(token) <= _JOINERS:
            # Whitespace or a hyphen inside a run keeps it going; the same
            # characters between two ordinary words are just punctuation.
            if seen:
                continue
            out.append(token)
        else:
            flush()
            out.append(token)
    flush()
    return "".join(out)


@dataclass(frozen=True, slots=True)
class Figure:
    """One number or date the agent said, and where it came from."""

    text: str
    traced: bool

    @property
    def digits(self) -> str:
        return _digits(self.text)


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _normalise(text: str) -> str:
    """Lower-case, with spoken numbers turned into digits.

    "eighty-five days" and "85 days" are the same claim, and the agent is told
    to say the first.
    """
    return _fold_spoken_numbers(text.lower())


def trace(spoken: str, tool_results: list[str]) -> list[Figure]:
    """Every figure in ``spoken``, marked with whether a tool supplied it.

    Matching is on digits rather than strings because the system prompt tells
    the agent to round for speech: "about seven thousand" against "7,200 pounds"
    is obedience, not invention. A figure counts as traced when the digits it is
    built from appear in a tool result, or when it is a plausible rounding of
    one that does.
    """
    if not spoken.strip():
        return []
    sources = " ".join(tool_results)
    source_digits = {_digits(match.group()) for match in _FIGURE.finditer(sources)}
    source_digits |= {_digits(match.group()) for match in _FIGURE.finditer(_normalise(sources))}
    source_digits.discard("")

    figures: list[Figure] = []
    for match in _FIGURE.finditer(_normalise(spoken)):
        said = match.group().strip()
        digits = _digits(said)
        if not digits:
            continue
        traced = digits in source_digits or any(
            _is_rounding(digits, candidate) for candidate in source_digits
        )
        figures.append(Figure(text=said, traced=traced))
    return figures


def _is_rounding(said: str, source: str) -> bool:
    """Whether ``said`` is a plausible spoken rounding of ``source``.

    Both are digit strings. "7" for "7200" is what "about seven thousand" looks
    like once the words are stripped, and refusing it would flag the agent for
    doing what it was told.
    """
    if not said or not source:
        return False
    # No prefix rule. It was here to let "seven" stand for "7,200" back when
    # number words were substituted one at a time, and it made every figure
    # beginning with a source digit look traced — "14" passed because a source
    # contained "1". Folding now yields whole values, so proportion is the only
    # test needed, and it is the one that means something.
    try:
        a, b = int(said), int(source)
    except ValueError:
        return False
    if a == 0 or b == 0:
        return False
    bigger, smaller = max(a, b), min(a, b)
    # Within 10% covers rounding to the nearest hundred or half-thousand.
    return (bigger - smaller) / bigger <= 0.1


def summarise(figures: list[Figure]) -> dict[str, object]:
    """The shape the dashboard shows: counts, and the untraced ones by name."""
    untraced = [f.text for f in figures if not f.traced]
    return {
        "spoken": len(figures),
        "traced": len(figures) - len(untraced),
        "untraced": untraced,
    }
