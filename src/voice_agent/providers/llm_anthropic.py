"""Anthropic LLM provider.

Three things about the current API shape matter here, and each is a trap that
fails loudly or silently rather than degrading:

**``temperature`` is rejected.** It was removed on Claude Opus 5 and returns a
400. The pipeline's ``llm_temperature`` setting is meaningful for Ollama and is
deliberately *not* forwarded here — a provider adapter that passed the whole
config through would break every request.

**Thinking is on by default.** Omitting the ``thinking`` parameter runs adaptive
thinking, which is the right default for hard reasoning and the wrong one for a
turn that must produce its first speakable clause inside 200ms. This adapter
sets an explicit effort level rather than inheriting the default.

**Disabling thinking is not free.** ``{"type": "disabled"}`` is accepted at
effort ``high`` or below, and is tempting for latency, but on this model it can
cause a tool call to be written into the visible text instead of emitted as a
structured block — the turn succeeds, the tool never runs, and nothing raises.
For a voice agent that would mean speaking a function call aloud. Measured on
this machine, disabling it is not even faster — 1047ms to first token with
thinking off against 871ms with adaptive thinking on, so the latency is network
round-trip and model time, not reasoning. With no speed argument left, the
default is adaptive thinking at ``low`` effort. Disabling remains available via
configuration, but nothing currently recommends it.

Note the wider figure: ~870ms to first token is on its own over the whole 800ms
first-audio budget, against 98ms for local Ollama. Single samples from one
location, so phase 5 measures this properly — but the cloud LLM is not
obviously the low-latency option it is usually assumed to be.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from voice_agent.config import Settings
from voice_agent.providers.base import Health, LlmDelta, Message, TextDelta, ToolCall

log = logging.getLogger(__name__)

#: Model families that accept adaptive thinking and the effort parameter.
#: Older models reject both with a 400, so the request shape has to depend on
#: which model is selected — sending the newer parameters to Haiku 4.5 fails
#: the request outright rather than being ignored.
_SUPPORTS_ADAPTIVE_THINKING = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)


class AnthropicLLM:
    """Streaming chat against the Anthropic Messages API."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-opus-5",
        max_tokens: int = 120,
        effort: str = "low",
        thinking: bool = True,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort
        self._thinking = thinking
        self._client: Any = None

    @classmethod
    def from_settings(cls, settings: Settings) -> AnthropicLLM:
        if settings.anthropic_api_key is None:
            raise RuntimeError("VA_ANTHROPIC_API_KEY is not set")
        return cls(
            settings.anthropic_api_key.get_secret_value(),
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            effort=settings.anthropic_effort,
            thinking=settings.anthropic_thinking,
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    def _request_kwargs(self) -> dict[str, Any]:
        """Build the request parameters.

        Deliberately does not include ``temperature``, ``top_p``, or ``top_k`` —
        all three are rejected on current models. Thinking and effort are added
        only for models that accept them; on an older model they are a 400
        rather than a no-op.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        if self._model.startswith(_SUPPORTS_ADAPTIVE_THINKING):
            kwargs["thinking"] = {"type": "adaptive"} if self._thinking else {"type": "disabled"}
            kwargs["output_config"] = {"effort": self._effort}
        return kwargs

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LlmDelta]:
        client = self._ensure_client()
        kwargs = self._request_kwargs()
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        kwargs["messages"] = _to_anthropic_messages(messages)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(tool) for tool in tools]

        # Tool arguments arrive as a stream of partial JSON fragments and are
        # only actionable once complete, so they are accumulated per block and
        # emitted whole at content_block_stop.
        pending: dict[int, dict[str, Any]] = {}

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                kind = getattr(event, "type", "")

                if kind == "content_block_start":
                    block = event.content_block
                    if getattr(block, "type", "") == "tool_use":
                        pending[event.index] = {
                            "id": block.id,
                            "name": block.name,
                            "json": "",
                        }

                elif kind == "content_block_delta":
                    delta = event.delta
                    delta_type = getattr(delta, "type", "")
                    if delta_type == "text_delta":
                        if delta.text:
                            yield TextDelta(delta.text)
                    elif delta_type == "input_json_delta":
                        entry = pending.get(event.index)
                        if entry is not None:
                            entry["json"] += delta.partial_json
                    # thinking_delta is intentionally dropped: it is reasoning,
                    # not speech, and must never reach the TTS queue.

                elif kind == "content_block_stop":
                    entry = pending.pop(event.index, None)
                    if entry is not None:
                        yield ToolCall(
                            id=entry["id"],
                            name=entry["name"],
                            arguments=_parse_arguments(entry["json"]),
                        )

    async def warmup(self) -> None:
        """Open the connection and validate credentials before the first call."""
        try:
            async for _ in self.stream([Message(role="user", content="Say OK.")]):
                break
            log.info("anthropic warmed model=%s", self._model)
        except Exception as exc:
            log.warning("anthropic warmup failed: %s", exc)

    async def health(self) -> Health:
        start = time.perf_counter()
        try:
            received = False
            async for delta in self.stream(
                [Message(role="user", content="Reply with the single word OK.")]
            ):
                if isinstance(delta, TextDelta):
                    received = True
                    break
            elapsed = (time.perf_counter() - start) * 1000.0
            if not received:
                return Health(ok=False, detail="no text returned", latency_ms=elapsed)
            return Health(ok=True, detail=self._model, latency_ms=elapsed)
        except Exception as exc:
            return Health(ok=False, detail=f"{type(exc).__name__}: {exc}")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert the neutral tool shape: the Messages API names the schema
    field ``input_schema``, not ``parameters``."""
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
    }


def _to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert to the Messages API shape.

    The tool loop is what makes this fiddly. A ``tool_result`` block is only
    valid if the preceding assistant turn contains the matching ``tool_use``, so
    an assistant message that made calls has to be rendered as content blocks
    rather than a bare string — otherwise the second turn of any tool-using
    conversation is rejected outright.
    """
    converted: list[dict[str, Any]] = []

    for message in messages:
        if message.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message.content,
            }
            # Results for calls made in the same assistant turn belong together
            # in one user message.
            if (
                converted
                and converted[-1]["role"] == "user"
                and isinstance(converted[-1]["content"], list)
            ):
                converted[-1]["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
            continue

        if message.role == "assistant" and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content.strip():
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            converted.append({"role": "assistant", "content": blocks})
            continue

        role = "assistant" if message.role == "assistant" else "user"
        if (
            converted
            and converted[-1]["role"] == role
            and isinstance(converted[-1]["content"], str)
        ):
            converted[-1]["content"] += "\n" + message.content
            continue
        converted.append({"role": role, "content": message.content})

    # The API requires the conversation to open with a user turn.
    while converted and converted[0]["role"] != "user":
        converted.pop(0)
    return converted


def _parse_arguments(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("could not parse tool arguments: %s", raw[:200])
        return {}
    return parsed if isinstance(parsed, dict) else {}
