"""FastAPI application.

Providers are built and warmed once at startup and shared across calls, because
warm-up is expensive enough to matter: the first CUDA inference was measured at
9.26s against 50ms warm, and a resident Piper process saves ~340ms on every
clause. Building either per call would hand that cost to the caller.

Per-call state lives in :class:`voice_agent.session.Session`, so sharing the
providers does not share anything a concurrent call could corrupt.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from voice_agent.agent.memory import ConversationStore
from voice_agent.agent.prompts import SYSTEM_PROMPT
from voice_agent.agent.runner import TurnSource, build_turn_source
from voice_agent.config import Settings, get_settings
from voice_agent.prompts_cache import PromptCache
from voice_agent.providers.registry import Providers, build_providers
from voice_agent.server.security import ephemeral_secret

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class AppState:
    """Everything shared across calls."""

    settings: Settings
    providers: Providers
    store: ConversationStore
    turns: TurnSource
    cache: PromptCache
    http: httpx.AsyncClient
    session_secret: str
    #: Currently selected model, as 'provider:model'.
    model_id: str = ""
    ready: bool = False
    #: Live sessions, for the health endpoint and for shutdown.
    active: int = 0


class CallInProgress(Exception):
    """Raised when an operation cannot run while a call is up.

    A dedicated type rather than RuntimeError: NotImplementedError is a
    RuntimeError subclass, so catching RuntimeError to mean 'busy' also
    caught 'unknown provider' and reported a conflict for a bad request.
    """


#: Cloud models offered by the picker. Kept short and opinionated rather than
#: enumerating everything: these are the three the benchmark actually compares.
CLOUD_MODELS: tuple[tuple[str, str], ...] = (
    ("claude-haiku-4-5", "Claude Haiku 4.5 — fastest, best value"),
    ("claude-sonnet-5", "Claude Sonnet 5 — strongest tool use"),
    ("claude-opus-5", "Claude Opus 5 — slowest here"),
)


async def available_models(ctx: AppState) -> list[dict[str, Any]]:
    """Every model that could be selected right now.

    Local models are discovered from Ollama rather than hard-coded, because
    which ones are installed is a property of the machine, not of this file.
    Cloud models are listed only when a key is present — offering a model that
    cannot possibly work is worse than not offering it.
    """
    models: list[dict[str, Any]] = []

    try:
        response = await ctx.http.get(f"{ctx.settings.ollama_url}/api/tags", timeout=4.0)
        response.raise_for_status()
        for entry in response.json().get("models", []):
            name = entry.get("name")
            if name:
                models.append(
                    {
                        "id": f"ollama:{name}",
                        "label": f"{name} — local, free",
                        "provider": "ollama",
                    }
                )
    except Exception as exc:
        log.debug("could not list Ollama models: %s", exc)

    if ctx.settings.anthropic_api_key is not None:
        models.extend(
            {"id": f"anthropic:{model}", "label": label, "provider": "anthropic"}
            for model, label in CLOUD_MODELS
        )

    return models


async def switch_model(ctx: AppState, model_id: str) -> str:
    """Rebuild the LLM and the agent layer around a different model.

    Refused while a call is in progress: the agent layer holds the model, and
    swapping it underneath a live turn would cancel that caller mid-sentence to
    serve a UI click.
    """
    if ctx.active:
        raise CallInProgress("a call is in progress; hang up before switching model")

    provider, _, model = model_id.partition(":")
    if not provider or not model:
        raise ValueError(f"expected 'provider:model', got {model_id!r}")

    from voice_agent.agent.runner import build_turn_source
    from voice_agent.providers.registry import build_llm_named

    new_llm = build_llm_named(ctx.settings, ctx.http, provider, model)
    old_llm = ctx.providers.llm

    ctx.providers.llm = new_llm
    ctx.turns = build_turn_source(ctx.settings, new_llm, ctx.store, system=SYSTEM_PROMPT)
    ctx.model_id = model_id

    # Warm before returning, so the first caller after a switch does not pay
    # model load — which on a cold Ollama model is measured in seconds.
    await new_llm.warmup()

    if old_llm is not new_llm:
        await old_llm.aclose()
    log.info("switched model to %s", model_id)
    return model_id


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings or get_settings()

    # One HTTP client for the process. A new client per request would pay TCP
    # and TLS setup on a path where the whole budget is 200ms.
    http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    secret = (
        settings.session_secret.get_secret_value()
        if settings.session_secret
        else ephemeral_secret()
    )
    if settings.session_secret is None:
        log.warning(
            "VA_SESSION_SECRET is not set; using a per-process secret. "
            "Sessions will not survive a restart."
        )

    providers = build_providers(settings, http)
    store = ConversationStore()
    turns = build_turn_source(settings, providers.llm, store, system=SYSTEM_PROMPT)
    state = AppState(
        settings=settings,
        providers=providers,
        store=store,
        turns=turns,
        cache=PromptCache(),
        http=http,
        session_secret=secret,
        model_id=f"{settings.llm_provider}:{_default_model(settings)}",
    )
    app.state.ctx = state

    async def prepare() -> None:
        await providers.warmup()
        # Must follow warm-up: the first synthesis on a cold Piper pays process
        # start, and pre-synthesising the fallback lines is the point at which
        # that cost is cheapest to absorb.
        await state.cache.build(providers.tts, settings.languages)
        state.ready = True
        log.info("ready (profile=%s)", settings.profile)

    warmup_task = asyncio.create_task(prepare(), name="warmup")

    try:
        yield
    finally:
        warmup_task.cancel()
        await asyncio.gather(warmup_task, return_exceptions=True)
        await providers.aclose()
        await http.aclose()
        log.info("shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="voice-agent",
        description="Low-latency agentic voice pipeline",
        lifespan=lifespan,
    )
    app.state.settings = settings or get_settings()
    # No CORS middleware. The demo page is served from this same origin, and the
    # wildcard-with-credentials configuration the previous gateway used is
    # rejected by the CORS spec anyway.
    from voice_agent.server.routes import router

    app.include_router(router)
    return app


def main() -> None:
    """Entry point for ``voice-agent``."""
    import uvicorn

    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="warning",  # our own logging is the useful one
    )


if __name__ == "__main__":  # pragma: no cover
    main()


def _default_model(settings: Settings) -> str:
    """The model the configured provider starts on."""
    if settings.llm_provider == "anthropic":
        return settings.anthropic_model
    if settings.llm_provider == "openai":
        return settings.openai_model
    return settings.ollama_model
