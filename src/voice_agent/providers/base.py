"""Provider protocols.

Three capabilities the pipeline needs, defined as :class:`typing.Protocol` so
implementations are structurally typed rather than inheriting from a base class.
Nothing in the pipeline imports a concrete provider; the registry resolves them
from configuration.

Every interface is **streaming-shaped**, including for providers that cannot
stream. Piper emits a whole clause at once and faster-whisper transcribes in one
shot, but they still yield through an async iterator, so swapping in a genuinely
streaming provider later is a configuration change rather than a rewrite of the
pipeline. Shaping the interface around today's non-streaming local stack would
bake a batch assumption into the one place it is most expensive to remove.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A completed tool invocation request.

    Emitted whole rather than streamed: a partially-parsed tool call is not
    actionable, and the pipeline has nothing useful to do with half an argument
    object. Text is streamed because it can be spoken incrementally; tool calls
    cannot.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of conversation, in a provider-neutral shape."""

    role: Role
    content: str
    #: Set on a tool result, naming the call it answers.
    tool_call_id: str | None = None
    name: str | None = None
    #: Tool calls this assistant turn made. Required for a multi-turn tool loop:
    #: the Messages API rejects a tool result that does not answer a tool_use
    #: block in the preceding assistant turn, so dropping these breaks the loop
    #: with a validation error rather than degrading.
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class TextDelta:
    """An incremental piece of assistant text."""

    text: str


LlmDelta = TextDelta | ToolCall


@dataclass(frozen=True, slots=True)
class Transcript:
    """The result of transcribing one utterance.

    ``confidence`` is normalised to 0..1 across providers so the pipeline's
    low-confidence branch does not need to know whether it is reading a Whisper
    average log-probability or a Deepgram score.
    """

    text: str
    confidence: float = 1.0
    language: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class Health:
    """Outcome of a provider health check."""

    ok: bool
    detail: str = ""
    latency_ms: float | None = None


@runtime_checkable
class ASR(Protocol):
    """Speech to text."""

    name: str

    async def transcribe(self, pcm: bytes) -> Transcript:
        """Transcribe one complete utterance of 16 kHz mono 16-bit PCM."""
        ...

    async def warmup(self) -> None:
        """Load models and compile kernels before the first call.

        Load-bearing rather than an optimisation: the first CUDA inference was
        measured at 9.26s against 50ms warm. A cold provider does not slow the
        first call down, it ruins it.
        """
        ...

    async def health(self) -> Health: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class TTS(Protocol):
    """Text to speech."""

    name: str

    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield 16 kHz mono 16-bit PCM for ``text``.

        Not an ``async def`` returning an iterator — an async generator function,
        so callers can ``async for`` directly and cancellation propagates into
        the generator on barge-in.
        """
        ...

    async def warmup(self) -> None: ...

    async def health(self) -> Health: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class LLM(Protocol):
    """Streaming chat completion with optional tool use."""

    name: str

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LlmDelta]:
        """Stream the assistant's reply as deltas.

        ``max_tokens`` overrides the configured budget for one call. The
        pipeline's budget is sized for two spoken sentences, which is far
        too small for a caller that needs the model to emit SQL — the
        statement is silently truncated mid-token and fails to parse.
        """
        ...

    async def warmup(self) -> None: ...

    async def health(self) -> Health: ...

    async def aclose(self) -> None: ...
