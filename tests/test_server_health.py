"""Server readiness should reflect provider health, not just startup completion."""

from __future__ import annotations

from types import SimpleNamespace

from voice_agent.providers.base import Health
from voice_agent.server.routes import _readiness


class Provider:
    def __init__(self, name: str) -> None:
        self.name = name


class Providers:
    def __init__(self, *, llm_ok: bool) -> None:
        self.asr = Provider("asr")
        self.tts = Provider("tts")
        self.llm = Provider("llm")
        self._llm_ok = llm_ok

    async def health(self) -> dict[str, Health]:
        return {
            "asr": Health(ok=True, detail="ready"),
            "tts": Health(ok=True, detail="ready"),
            "llm": Health(ok=self._llm_ok, detail="ready" if self._llm_ok else "offline"),
        }


async def test_readiness_fails_when_the_llm_is_offline() -> None:
    ctx = SimpleNamespace(
        ready=True,
        cache=SimpleNamespace(ready=True),
        providers=Providers(llm_ok=False),
        turns=SimpleNamespace(name="langgraph"),
    )

    ok, body = await _readiness(ctx)  # type: ignore[arg-type]

    assert ok is False
    assert body["ok"] is False
    assert "llm: offline" in str(body["detail"])


async def test_readiness_passes_when_all_providers_are_healthy() -> None:
    ctx = SimpleNamespace(
        ready=True,
        cache=SimpleNamespace(ready=True),
        providers=Providers(llm_ok=True),
        turns=SimpleNamespace(name="langgraph"),
    )

    ok, body = await _readiness(ctx)  # type: ignore[arg-type]

    assert ok is True
    assert body["ok"] is True
