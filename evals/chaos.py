"""Fault injection.

The failure paths in the pipeline are specified behaviour, and specified
behaviour that is never exercised is a comment. These wrappers make each
failure happen on demand so the recovery can be asserted rather than assumed:
an ASR that returns nothing, a model that stalls past its deadline, a
synthesiser that dies mid-utterance.

Each wrapper satisfies the same protocol as the provider it wraps, so the
pipeline cannot tell it is being sabotaged — which is the point. A test that
reaches inside the pipeline to trigger a failure proves only that the test can
reach inside the pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from voice_agent.providers.base import (
    ASR,
    LLM,
    TTS,
    Health,
    LlmDelta,
    Message,
    Transcript,
)

log = logging.getLogger(__name__)


class BlankingASR:
    """Returns nothing for the first ``blanks`` utterances.

    Simulates a caller who cannot be heard: a bad line, a hand over the
    microphone, a lorry going past. The pipeline should ask them to repeat
    without advancing the turn or writing anything into history.
    """

    def __init__(self, inner: ASR, blanks: int) -> None:
        self._inner = inner
        self._remaining = blanks
        self.name = f"{inner.name}+blanking"
        self.fired = 0

    async def transcribe(self, pcm: bytes) -> Transcript:
        if self._remaining > 0:
            self._remaining -= 1
            self.fired += 1
            log.info("[chaos] blanking transcript (%d left)", self._remaining)
            return Transcript(text="", confidence=0.0)
        return await self._inner.transcribe(pcm)

    async def warmup(self) -> None:
        await self._inner.warmup()

    async def health(self) -> Health:
        return await self._inner.health()

    async def aclose(self) -> None:
        await self._inner.aclose()


class StallingLLM:
    """Delays the first token on a chosen turn, past the filler deadline.

    The agent should cover the gap with its holding line rather than leaving
    the caller in silence — silence being what makes callers start talking,
    which trips barge-in, which cancels the recovery.
    """

    def __init__(self, inner: LLM, *, on_turn: int, seconds: float) -> None:
        self._inner = inner
        self._on_turn = on_turn
        self._seconds = seconds
        self._turn = 0
        self.name = f"{inner.name}+stalling"
        self.fired = 0

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LlmDelta]:
        self._turn += 1
        if self._turn == self._on_turn:
            self.fired += 1
            log.info("[chaos] stalling %.1fs before first token", self._seconds)
            await asyncio.sleep(self._seconds)
        async for delta in self._inner.stream(
            messages, system=system, tools=tools, max_tokens=max_tokens
        ):
            yield delta

    async def warmup(self) -> None:
        await self._inner.warmup()

    async def health(self) -> Health:
        return await self._inner.health()

    async def aclose(self) -> None:
        await self._inner.aclose()


class DyingLLM:
    """Raises partway through generation on a chosen turn.

    The pipeline should salvage buffered text if it stops somewhere speakable
    and otherwise fall back to its error line — never simply stop talking.
    """

    def __init__(self, inner: LLM, *, on_turn: int, after_deltas: int = 3) -> None:
        self._inner = inner
        self._on_turn = on_turn
        self._after = after_deltas
        self._turn = 0
        self.name = f"{inner.name}+dying"
        self.fired = 0

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LlmDelta]:
        self._turn += 1
        failing = self._turn == self._on_turn
        seen = 0
        async for delta in self._inner.stream(
            messages, system=system, tools=tools, max_tokens=max_tokens
        ):
            yield delta
            seen += 1
            if failing and seen >= self._after:
                self.fired += 1
                log.info("[chaos] killing the stream after %d deltas", seen)
                raise RuntimeError("simulated upstream failure")

    async def warmup(self) -> None:
        await self._inner.warmup()

    async def health(self) -> Health:
        return await self._inner.health()

    async def aclose(self) -> None:
        await self._inner.aclose()


class FailingTTS:
    """Refuses to synthesise on a chosen clause.

    A missing clause should cost the caller a phrase, not the call.
    """

    def __init__(self, inner: TTS, *, on_clause: int) -> None:
        self._inner = inner
        self._on_clause = on_clause
        self._clause = 0
        self.name = f"{inner.name}+failing"
        self.fired = 0

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self._clause += 1
        if self._clause == self._on_clause:
            self.fired += 1
            log.info("[chaos] refusing to synthesise clause %d", self._clause)
            raise RuntimeError("simulated synthesis failure")
        async for chunk in self._inner.synthesize(text):
            yield chunk

    async def warmup(self) -> None:
        await self._inner.warmup()

    async def health(self) -> Health:
        return await self._inner.health()

    async def aclose(self) -> None:
        await self._inner.aclose()


def apply(faults: dict[str, Any], asr: ASR, tts: TTS, llm: LLM) -> tuple[ASR, TTS, LLM, list[Any]]:
    """Wrap providers according to a scenario's fault declaration."""
    injected: list[Any] = []

    if blanks := faults.get("asr_blank"):
        asr = BlankingASR(asr, int(blanks))
        injected.append(asr)
    if turn := faults.get("llm_stall_turn"):
        llm = StallingLLM(
            llm, on_turn=int(turn), seconds=float(faults.get("llm_stall_seconds", 2.0))
        )
        injected.append(llm)
    if turn := faults.get("llm_die_turn"):
        llm = DyingLLM(llm, on_turn=int(turn))
        injected.append(llm)
    if clause := faults.get("tts_fail_clause"):
        tts = FailingTTS(tts, on_clause=int(clause))
        injected.append(tts)

    return asr, tts, llm, injected
