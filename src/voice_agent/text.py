"""Turning a token stream into speakable clauses.

The pipeline cannot wait for a complete response before speaking — that would
put the whole generation on the critical path. It also cannot hand TTS one token
at a time, because prosody comes from synthesising a phrase as a unit.

So text is cut at the earliest point that is both *speakable* and *long enough
to sound natural*. The budget allows 80ms from first token to first speakable
clause, which rules out waiting for a sentence boundary: "I can see three open
orders on your account, and the most recent one shipped on Tuesday" is one
sentence and several seconds of generation. Cutting at the comma gets audio
moving while the rest is still being written.
"""

from __future__ import annotations

import re
import time

#: End of a sentence, allowing for a closing quote or bracket after the stop.
_SENTENCE_END = re.compile(r"[.!?]['\")\]]*(?=\s|$)")

#: Internal clause break. Speakable, and usually a natural prosodic pause.
_CLAUSE_END = re.compile(r"[,;:](?=\s)")

#: Markdown and other artefacts that must never reach a speech synthesiser.
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MARKDOWN_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(.+?)\1")
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_LIST_BULLET = re.compile(r"^\s*[-*+]\s+", re.M)
_LEADING_LABEL = re.compile(r"^\s*(assistant|agent|reply|response)\s*[:\-]\s*", re.I)
_REPEATED_PUNCT = re.compile(r"([.!?])\1{1,}")
_WHITESPACE = re.compile(r"\s+")

#: A tool call the model wrote as prose instead of emitting as a structured
#: block. Smaller models do this regularly, and a voice agent that reads
#: `(escalate_to_human {"reason": ...})` down the phone is worse than one
#: that says nothing — so these are removed before anything is synthesised.
#: An identifier immediately followed by a JSON object, with or without
#: surrounding parentheses: ``escalate_to_human {"reason": ...}``,
#: ``lookup_customer({"name": ...})``. Requiring the braces is what keeps this
#: from eating ordinary prose.
_LEAKED_TOOL_CALL = re.compile(
    r"\(?\s*[a-z_][a-z0-9_]{2,39}\s*\(?\s*\{[^{}]{0,400}\}\s*\)?\s*\)?",
    re.I | re.S,
)
#: The XML-ish wrappers some models use for the same thing.
_TOOL_TAGS = re.compile(
    r"</?(?:tool_call|function_call|tool_use|thinking|invoke|parameter)[^>]*>",
    re.I,
)

#: Hard ceiling on one synthesis request. Piper slows markedly on very long
#: input, and nothing the agent says in a phone call should approach this.
MAX_CLAUSE_CHARS = 600

#: Characters a synthesiser mispronounces or reads out literally. Written as
#: escapes rather than literals because an em dash, an en dash and a hyphen are
#: indistinguishable in source, and getting the wrong one silently does nothing.
_SPOKEN_SUBSTITUTIONS = {
    chr(0x2014): ", ",  # em dash
    chr(0x2013): ", ",  # en dash
    chr(0x2022): " ",  # bullet
    chr(0x2026): ". ",  # ellipsis
    chr(0x00A0): " ",  # non-breaking space
}


def clean_for_speech(text: str) -> str:
    """Strip anything a synthesiser would read out literally or choke on.

    Even with a system prompt asking for plain speech, models emit the
    occasional bullet, bold span, or ``Assistant:`` prefix. Piper pronounces
    asterisks.
    """
    if not text:
        return ""
    cleaned = _CODE_FENCE.sub(" ", text)
    cleaned = _TOOL_TAGS.sub(" ", cleaned)
    cleaned = _LEAKED_TOOL_CALL.sub(" ", cleaned)
    cleaned = _INLINE_CODE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_HEADING.sub("", cleaned)
    cleaned = _LIST_BULLET.sub("", cleaned)
    cleaned = _MARKDOWN_EMPHASIS.sub(r"\2", cleaned)
    cleaned = _LEADING_LABEL.sub("", cleaned)
    for character, replacement in _SPOKEN_SUBSTITUTIONS.items():
        cleaned = cleaned.replace(character, replacement)
    cleaned = _REPEATED_PUNCT.sub(r"\1", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return cleaned[:MAX_CLAUSE_CHARS]


class ClauseAssembler:
    """Accumulates streamed text and emits clauses when they are ready.

    Three ways a clause becomes ready, in priority order:

    1. A sentence ends and the buffer is long enough to be worth speaking.
    2. An internal clause break (comma, semicolon, colon) with enough text
       behind it — the mechanism that makes first-audio fast on a long sentence.
    3. Nothing has been emitted for ``flush_after_ms`` and there is enough text
       to say. A model that produces a long unpunctuated run should not leave
       the caller in silence waiting for a comma that never comes.
    """

    __slots__ = ("_buffer", "_flush_after_s", "_last_emit", "_min_chars")

    def __init__(self, *, min_chars: int = 12, flush_after_ms: int = 400) -> None:
        self._min_chars = max(1, min_chars)
        self._flush_after_s = flush_after_ms / 1000.0
        self._buffer = ""
        # Anchored on the first push rather than at construction, so the timer
        # shares whatever clock the caller uses. Seeding it from time.monotonic()
        # here makes an injected clock incomparable with it, and the time-based
        # flush then silently never fires.
        self._last_emit: float | None = None

    def push(self, text: str, *, now: float | None = None) -> list[str]:
        """Add streamed text; return any clauses that are now speakable."""
        moment = time.monotonic() if now is None else now
        if self._last_emit is None:
            self._last_emit = moment
        self._buffer += text

        ready: list[str] = []
        while (index := self._boundary()) is not None:
            clause = self._buffer[:index].strip()
            self._buffer = self._buffer[index:].lstrip()
            if clause:
                ready.append(clause)
                self._last_emit = moment

        if not ready and self._should_time_flush(moment):
            clause = self._buffer.strip()
            self._buffer = ""
            self._last_emit = moment
            ready.append(clause)

        return ready

    def _boundary(self) -> int | None:
        """Index just past the earliest usable break, or ``None``."""
        for pattern in (_SENTENCE_END, _CLAUSE_END):
            for match in pattern.finditer(self._buffer):
                end = match.end()
                if len(self._buffer[:end].strip()) >= self._min_chars:
                    return end
        return None

    def _should_time_flush(self, now: float) -> bool:
        if self._last_emit is None:
            return False
        if len(self._buffer.strip()) < self._min_chars:
            return False
        return (now - self._last_emit) >= self._flush_after_s

    def flush(self) -> str | None:
        """Emit whatever is left once the stream has ended.

        No minimum length here — the model has stopped, so a short trailing
        fragment is the whole rest of the answer, not a premature cut.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        self._last_emit = None
        return remaining or None

    @property
    def pending(self) -> str:
        return self._buffer

    def ends_at_boundary(self) -> bool:
        """Whether the buffered text stops somewhere it could be spoken.

        Used when an LLM stream dies mid-response: text that ends on a clause
        boundary can be salvaged and spoken, whereas a fragment cut mid-word
        would sound like a fault.
        """
        stripped = self._buffer.rstrip()
        return bool(stripped) and stripped[-1] in ".!?,;:"
