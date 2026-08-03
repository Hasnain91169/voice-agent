"""Build providers from configuration.

The single place that knows which concrete classes exist. The pipeline depends
on the protocols in :mod:`voice_agent.providers.base` and never imports an
implementation, so adding a provider means adding a branch here rather than
touching the turn loop.

Imports are deferred into each branch because the optional extras are genuinely
optional: a local-only install has no ``anthropic`` package, and a cloud-only
deployment has no ``faster_whisper``. A module-level import of either would make
the unused half a hard dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from voice_agent.config import Settings
from voice_agent.providers.base import ASR, LLM, TTS, Health

log = logging.getLogger(__name__)


@dataclass
class Providers:
    """The three providers a session needs, plus lifecycle for them."""

    asr: ASR
    tts: TTS
    llm: LLM

    async def warmup(self) -> None:
        """Warm every provider concurrently.

        Concurrently because they are independent and startup is on the critical
        path for the first call: warming serially would mean waiting for the
        Whisper CUDA kernel compile *and then* the Piper process start.
        """
        import asyncio

        await asyncio.gather(self.asr.warmup(), self.tts.warmup(), self.llm.warmup())

    async def health(self) -> dict[str, Health]:
        import asyncio

        asr, tts, llm = await asyncio.gather(
            self.asr.health(), self.tts.health(), self.llm.health()
        )
        return {"asr": asr, "tts": tts, "llm": llm}

    async def aclose(self) -> None:
        import asyncio

        await asyncio.gather(
            self.asr.aclose(),
            self.tts.aclose(),
            self.llm.aclose(),
            return_exceptions=True,
        )


def build_asr(settings: Settings) -> ASR:
    if settings.asr_provider == "faster_whisper":
        from voice_agent.providers.asr_faster_whisper import FasterWhisperASR

        return FasterWhisperASR.from_settings(settings)
    raise NotImplementedError(f"ASR provider {settings.asr_provider!r} is not implemented yet")


def build_tts(settings: Settings) -> TTS:
    if settings.tts_provider == "piper":
        from voice_agent.providers.tts_piper import PiperTTS

        return PiperTTS.from_settings(settings)
    raise NotImplementedError(f"TTS provider {settings.tts_provider!r} is not implemented yet")


def build_llm(settings: Settings, client: httpx.AsyncClient) -> LLM:
    if settings.llm_provider == "ollama":
        from voice_agent.providers.llm_ollama import OllamaLLM

        return OllamaLLM.from_settings(settings, client)
    if settings.llm_provider == "anthropic":
        from voice_agent.providers.llm_anthropic import AnthropicLLM

        return AnthropicLLM.from_settings(settings)
    raise NotImplementedError(f"LLM provider {settings.llm_provider!r} is not implemented yet")


def build_llm_named(
    settings: Settings, client: httpx.AsyncClient, provider: str, model: str
) -> LLM:
    """Build a specific model, ignoring the configured default.

    Used by the runtime model picker. Kept separate from :func:`build_llm` so
    that switching model cannot accidentally become the way the default is
    chosen — the profile still owns that.
    """
    if provider == "ollama":
        from voice_agent.providers.llm_ollama import OllamaLLM

        return OllamaLLM(
            client,
            base_url=settings.ollama_url,
            model=model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            keep_alive=settings.ollama_keep_alive,
        )
    if provider == "anthropic":
        from voice_agent.providers.llm_anthropic import AnthropicLLM

        if settings.anthropic_api_key is None:
            raise RuntimeError("VA_ANTHROPIC_API_KEY is not set")
        return AnthropicLLM(
            settings.anthropic_api_key.get_secret_value(),
            model=model,
            max_tokens=settings.llm_max_tokens,
            effort=settings.anthropic_effort,
            thinking=settings.anthropic_thinking,
        )
    raise NotImplementedError(f"unknown LLM provider {provider!r}")


def build_providers(settings: Settings, client: httpx.AsyncClient) -> Providers:
    """Resolve all three providers for the configured profile."""
    missing = settings.missing_credentials()
    if missing:
        raise RuntimeError(f"Missing credentials for the selected providers: {', '.join(missing)}")
    providers = Providers(
        asr=build_asr(settings),
        tts=build_tts(settings),
        llm=build_llm(settings, client),
    )
    log.info(
        "providers: asr=%s tts=%s llm=%s (profile=%s)",
        providers.asr.name,
        providers.tts.name,
        providers.llm.name,
        settings.profile,
    )
    return providers
