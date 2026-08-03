"""Per-call state.

Everything here is scoped to one call and owned by one object, which is the
fix for the defect that made the previous implementation unusable under load:
its echo-control state lived in module-level globals, so two concurrent calls
muted each other's microphones.

**This object deliberately does not hold the conversation.** It owns transport
and turn control — thresholds, echo timing, the frame channel, playback
progress, metrics — and nothing the agent reasons over. Message history lives
behind :mod:`voice_agent.agent.memory`, and the only thing crossing between
them is a ``thread_id`` string plus a single one-directional call at the end of
each turn. Two stores that both believe they hold the conversation is a bug
waiting to happen; one store and a pointer is not.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from voice_agent.audio.vad import EchoGate, Thresholds
from voice_agent.config import BYTES_PER_FRAME, FRAME_MS, Settings
from voice_agent.rx import FrameChannel
from voice_agent.transports.base import Transport


@dataclass
class SpokenTracker:
    """Maps audio actually sent back to the words the caller actually heard.

    When the caller interrupts, the agent has generated more than it said. If
    the full generated text were committed to history, the agent would spend the
    rest of the call believing it had told the caller things they never heard —
    referring back to them, not repeating them, answering follow-ups that were
    never asked. Tracking bytes per clause makes the committed text match the
    audio.
    """

    #: (text, byte length of its audio), in playback order.
    clauses: list[tuple[str, int]] = field(default_factory=list)

    def add(self, text: str, pcm_bytes: int) -> None:
        if text and pcm_bytes > 0:
            self.clauses.append((text, pcm_bytes))

    @property
    def full_text(self) -> str:
        return " ".join(text for text, _ in self.clauses).strip()

    @property
    def total_bytes(self) -> int:
        return sum(length for _, length in self.clauses)

    def spoken(self, played_bytes: int) -> str:
        """The text corresponding to the first ``played_bytes`` of audio.

        A clause cut partway through is truncated at a word boundary rather
        than mid-word: the point is an honest record of what was heard, and
        half a word is neither honest nor readable.
        """
        if played_bytes >= self.total_bytes:
            return self.full_text

        parts: list[str] = []
        remaining = played_bytes
        for text, length in self.clauses:
            if remaining >= length:
                parts.append(text)
                remaining -= length
                continue
            if remaining > 0:
                fraction = remaining / length
                cutoff = int(len(text) * fraction)
                partial = text[:cutoff].rsplit(" ", 1)[0] if cutoff < len(text) else text
                if partial.strip():
                    parts.append(partial.strip())
            break
        return " ".join(parts).strip()


@dataclass
class TurnMetrics:
    """Per-turn timings, recorded against the published budget."""

    endpoint_ms: float = 0.0
    asr_ms: float = 0.0
    llm_first_token_ms: float = 0.0
    first_clause_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    first_audio_ms: float = 0.0
    barged_in: bool = False
    #: Language this turn was heard and answered in, detected per utterance.
    language: str = "en"
    #: Characters generated versus characters the caller actually heard.
    #: Equal on an uninterrupted turn. After a barge-in they must differ, and
    #: that gap is the only externally visible proof that history recorded
    #: what was heard rather than what was produced.
    generated_chars: int = 0
    spoken_chars: int = 0
    #: Which failure paths fired, if any.
    events: list[str] = field(default_factory=list)

    def note(self, event: str) -> None:
        self.events.append(event)

    def summary(self) -> str:
        parts = [
            f"endpoint={self.endpoint_ms:.0f}",
            f"asr={self.asr_ms:.0f}",
            f"ttft={self.llm_first_token_ms:.0f}",
            f"clause={self.first_clause_ms:.0f}",
            f"tts={self.tts_first_chunk_ms:.0f}",
            f"first_audio={self.first_audio_ms:.0f}",
        ]
        if self.language != "en":
            parts.append(f"lang={self.language}")
        if self.barged_in:
            parts.append(f"barged_in({self.spoken_chars}/{self.generated_chars} chars)")
        if self.events:
            parts.append("events=" + ",".join(self.events))
        return " ".join(parts)


@dataclass
class Session:
    """Transport and turn-control state for a single call."""

    transport: Transport
    channel: FrameChannel
    settings: Settings
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    #: Pointer into the conversation store. Never the conversation itself.
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    thresholds: Thresholds = field(
        default_factory=lambda: Thresholds(floor=0.0, start=12.0, stop=8.0)
    )
    echo: EchoGate = field(init=False)

    #: True while playback is running. The barge-in detector is only armed here.
    speaking: bool = False
    #: Bytes of the current turn's audio handed to the transport.
    played_bytes: int = 0
    started_at: float = field(default_factory=time.monotonic)
    turns: int = 0
    #: Completed turns, in order. Kept so an evaluation harness can assert on
    #: what the pipeline did rather than on what it logged — barge-in was
    #: implemented, logged, and untested for exactly as long as the only
    #: evidence it fired was a line in stdout.
    history: list[TurnMetrics] = field(default_factory=list)

    def __post_init__(self) -> None:
        # A browser negotiates echo cancellation, so it needs far less of the
        # guard window that protects a phone line from the agent's own voice.
        # Not none, though: browser AEC is tuned for the speaker-to-microphone
        # path and leaks on laptop speakers at volume, which is heard as the
        # agent interrupting itself.
        guard = (
            self.settings.echo_cancelled_guard_ms
            if self.transport.echo_cancelled
            else self.settings.post_tts_guard_ms
        )
        self.echo = EchoGate(
            guard_ms=guard,
            raised_factor=self.settings.echo_threshold_factor,
        )

    def begin_playback(self) -> None:
        self.speaking = True
        self.played_bytes = 0
        self.echo.on_playback_start()

    def end_playback(self) -> None:
        self.speaking = False
        self.echo.on_playback_end(at=time.monotonic())

    @property
    def played_ms(self) -> int:
        return self.played_bytes // BYTES_PER_FRAME * FRAME_MS

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"<Session {self.id} transport={self.transport.name} "
            f"turns={self.turns} speaking={self.speaking}>"
        )
