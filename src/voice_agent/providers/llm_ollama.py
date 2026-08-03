"""Ollama LLM provider.

Local, free, and measured at 107ms to first token for ``qwen2.5:7b`` — well
inside the 200ms the budget allows. Time-to-first-token is the only latency that
matters here: the pipeline starts speaking as soon as it has a clause, so total
generation time is almost irrelevant to what the caller experiences.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from voice_agent.config import Settings
from voice_agent.providers.base import Health, LlmDelta, Message, TextDelta, ToolCall

log = logging.getLogger(__name__)


class OllamaLLM:
    """Streaming chat against a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.4,
        max_tokens: int = 120,
        keep_alive: str = "5m",
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._keep_alive = keep_alive

    @classmethod
    def from_settings(cls, settings: Settings, client: httpx.AsyncClient) -> OllamaLLM:
        return cls(
            client,
            base_url=settings.ollama_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            keep_alive=settings.ollama_keep_alive,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LlmDelta]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _to_ollama_messages(messages, system),
            "stream": True,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": self._temperature,
                "num_predict": max_tokens or self._max_tokens,
            },
        }
        if tools:
            payload["tools"] = [_to_ollama_tool(tool) for tool in tools]

        async with self._client.stream(
            "POST", f"{self._base_url}/api/chat", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = event.get("message") or {}
                for call in message.get("tool_calls") or []:
                    parsed = _parse_tool_call(call)
                    if parsed is not None:
                        yield parsed

                text = message.get("content") or ""
                if text:
                    yield TextDelta(text)

                if event.get("done"):
                    return

    async def warmup(self) -> None:
        """Force the model resident so the first caller doesn't pay the load."""
        try:
            async for _ in self.stream([Message(role="user", content="Say OK.")]):
                break
            log.info("ollama warmed model=%s", self._model)
        except Exception as exc:
            log.warning("ollama warmup failed: %s", exc)

    async def health(self) -> Health:
        start = time.perf_counter()
        try:
            response = await self._client.get(f"{self._base_url}/api/tags", timeout=4.0)
            response.raise_for_status()
            installed = {m.get("name", "") for m in response.json().get("models", [])}
            elapsed = (time.perf_counter() - start) * 1000.0
            if self._model not in installed:
                return Health(
                    ok=False,
                    detail=f"model {self._model!r} not installed (`ollama pull {self._model}`)",
                    latency_ms=elapsed,
                )
            return Health(ok=True, detail=self._model, latency_ms=elapsed)
        except Exception as exc:
            return Health(ok=False, detail=f"{type(exc).__name__}: {exc}")

    async def aclose(self) -> None:
        """The HTTP client is owned by the application, not by this provider."""


def _to_ollama_messages(messages: Sequence[Message], system: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name:
            entry["name"] = message.name
        if message.tool_calls:
            # Without these the tool results that follow answer nothing, and the
            # model re-requests the same lookup on the next round.
            entry["tool_calls"] = [
                {
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                }
                for call in message.tool_calls
            ]
        out.append(entry)
    return out


def _to_ollama_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert the neutral tool shape into Ollama's OpenAI-style envelope."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _parse_tool_call(call: dict[str, Any]) -> ToolCall | None:
    function = call.get("function") or {}
    name = function.get("name")
    if not name:
        return None
    arguments = function.get("arguments")
    # Ollama sends an object for most models and a JSON string for some.
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCall(id=call.get("id") or name, name=name, arguments=arguments)
