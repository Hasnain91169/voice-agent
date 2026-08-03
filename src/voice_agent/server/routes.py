"""HTTP and WebSocket routes."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Query, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse

from voice_agent.pipeline import run_call
from voice_agent.providers.base import ASR, LLM, TTS
from voice_agent.rx import FrameChannel, RxPump
from voice_agent.server.app import (
    STATIC_DIR,
    AppState,
    CallInProgress,
    available_models,
    switch_model,
)
from voice_agent.server.security import issue_token, verify_token
from voice_agent.session import Session
from voice_agent.transports.browser import BrowserTransport

log = logging.getLogger(__name__)

router = APIRouter()


def _ctx(request: Request) -> AppState:
    return request.app.state.ctx  # type: ignore[no-any-return]


@router.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "demo.html")


@router.get("/api/eval-summary")
def eval_summary() -> JSONResponse:
    """The last recorded eval run, or nothing.

    Written by ``python -m evals`` and committed, so the dashboard can show a
    real result rather than a live score it has no way to compute. Returning
    404 when absent is deliberate: the panel then says no run is recorded,
    which is true, instead of showing zeros that look like failures.
    """
    path = STATIC_DIR / "eval-summary.json"
    if not path.is_file():
        return JSONResponse({"detail": "no eval run recorded"}, status_code=404)
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@router.get("/demo")
def demo() -> FileResponse:
    return FileResponse(STATIC_DIR / "demo.html")


@router.get("/favicon.svg")
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Overall readiness.

    Reports 503 until every provider is warm. A cold provider does not make the
    first call slow, it makes it fail — so refusing traffic until warm is the
    honest behaviour, not a nicety.
    """
    ctx = _ctx(request)
    body = {
        "ok": ctx.ready,
        "profile": str(ctx.settings.profile),
        "providers": {
            "asr": ctx.providers.asr.name,
            "tts": ctx.providers.tts.name,
            "llm": ctx.providers.llm.name,
            "agent": ctx.turns.name,
        },
        "model": ctx.model_id,
        "prompt_cache": {
            "ready": ctx.cache.ready,
            "missing": ctx.cache.missing,
        },
        "active_sessions": ctx.active,
    }
    return JSONResponse(body, status_code=200 if ctx.ready else 503)


@router.get("/health/{component}")
async def health_component(request: Request, component: str) -> JSONResponse:
    """Per-provider check, so a failure names the component that caused it."""
    ctx = _ctx(request)
    providers: dict[str, ASR | TTS | LLM] = {
        "asr": ctx.providers.asr,
        "tts": ctx.providers.tts,
        "llm": ctx.providers.llm,
    }
    provider = providers.get(component)
    if provider is None:
        return JSONResponse({"error": f"unknown component {component!r}"}, 404)

    result = await provider.health()
    return JSONResponse(
        {
            "ok": result.ok,
            "component": component,
            "provider": provider.name,
            "detail": result.detail,
            "latency_ms": result.latency_ms,
        },
        status_code=200 if result.ok else 503,
    )


@router.get("/api/models")
async def get_models(request: Request) -> JSONResponse:
    """Models the picker can offer, and which one is live."""
    ctx = _ctx(request)
    return JSONResponse(
        {
            "current": ctx.model_id,
            "busy": ctx.active > 0,
            "models": await available_models(ctx),
        }
    )


@router.post("/api/model")
async def set_model(request: Request) -> JSONResponse:
    """Switch the model the agent runs on."""
    ctx = _ctx(request)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "expected a JSON body"}, status_code=400)

    model_id = str(body.get("id", ""))
    try:
        current = await switch_model(ctx, model_id)
    except CallInProgress as exc:
        # A conflict with current state, not a malformed request.
        return JSONResponse({"error": str(exc)}, status_code=409)
    except (ValueError, NotImplementedError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        log.exception("could not switch model")
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=502)
    return JSONResponse({"current": current})


@router.post("/api/session")
async def create_session(request: Request) -> JSONResponse:
    """Issue a short-lived token for the media socket."""
    ctx = _ctx(request)
    if not ctx.ready:
        return JSONResponse({"error": "warming up"}, status_code=503)
    token = issue_token(ctx.session_secret, ttl_s=ctx.settings.session_token_ttl_s)
    return JSONResponse(
        {
            "token": token,
            "expires_in": ctx.settings.session_token_ttl_s,
            "sample_rate": 16_000,
            "frame_ms": 20,
        }
    )


@router.websocket("/ws/browser")
async def ws_browser(websocket: WebSocket, token: str = Query(default="")) -> None:
    """Media socket for the browser demo."""
    ctx: AppState = websocket.app.state.ctx

    if not ctx.ready:
        await websocket.close(code=1013, reason="warming up")
        return
    # Checked before accepting: an unauthenticated peer should never reach the
    # point of being able to send audio.
    if not verify_token(ctx.session_secret, token):
        log.warning("rejected websocket with invalid token")
        await websocket.close(code=1008, reason="invalid or expired token")
        return

    await websocket.accept()
    transport = BrowserTransport(websocket)
    channel = FrameChannel()
    session = Session(transport=transport, channel=channel, settings=ctx.settings)
    ctx.active += 1
    log.info("[%s] connected (%d active)", session.id, ctx.active)

    pump = asyncio.create_task(RxPump(transport, channel).run(), name="rx")
    try:
        await run_call(
            session,
            ctx.providers,
            ctx.turns,
            ctx.cache,
            ctx.settings,
            # The browser is the only transport that has anywhere to put these.
            events=transport.send_event,
        )
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)
        await transport.close()
        ctx.active -= 1
        log.info("[%s] disconnected (%d active)", session.id, ctx.active)
