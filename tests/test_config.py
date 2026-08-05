"""Tests for profile resolution and startup credential checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_agent.config import LatencyBudget, Profile, Settings
from voice_agent.providers.tts_piper import PiperTTS


def build(**overrides: object) -> Settings:
    """Settings isolated from the developer's own environment.

    Explicit constructor arguments outrank environment variables in
    pydantic-settings, so these tests behave the same on a machine with real API
    keys exported as on a bare CI runner.
    """
    defaults: dict[str, object] = {
        "openai_api_key": None,
        "anthropic_api_key": None,
        "deepgram_api_key": None,
        "elevenlabs_api_key": None,
    }
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestProfiles:
    def test_default_profile_needs_no_api_keys(self) -> None:
        # The repo must be clonable and runnable without signing up for anything.
        settings = build()
        assert settings.profile is Profile.LOCAL_ZERO_COST
        assert settings.asr_provider == "faster_whisper"
        assert settings.tts_provider == "piper"
        assert settings.llm_provider == "ollama"
        assert settings.missing_credentials() == []

    def test_low_latency_profile_selects_streaming_providers(self) -> None:
        settings = build(profile=Profile.LOW_LATENCY)
        assert settings.asr_provider == "deepgram"
        assert settings.tts_provider == "elevenlabs"
        assert settings.llm_provider == "openai"

    def test_explicit_provider_outranks_the_profile(self) -> None:
        # Mixing local ASR with a cloud LLM is a legitimate combination while
        # iterating, so a profile is a starting point rather than a straitjacket.
        settings = build(profile=Profile.LOW_LATENCY, asr_provider="faster_whisper")
        assert settings.asr_provider == "faster_whisper"
        assert settings.tts_provider == "elevenlabs"


class TestCredentialCheck:
    def test_reports_every_missing_key_for_the_selected_providers(self) -> None:
        # Reported at startup so a misconfigured deploy fails at the health check
        # rather than halfway through the first live call.
        missing = build(profile=Profile.LOW_LATENCY).missing_credentials()
        assert set(missing) == {
            "VA_DEEPGRAM_API_KEY",
            "VA_ELEVENLABS_API_KEY",
            "VA_OPENAI_API_KEY",
        }

    def test_only_checks_providers_actually_in_use(self) -> None:
        missing = build(profile=Profile.LOW_LATENCY, llm_provider="ollama").missing_credentials()
        assert "VA_OPENAI_API_KEY" not in missing

    def test_anthropic_key_is_checked_when_selected(self) -> None:
        missing = build(llm_provider="anthropic").missing_credentials()
        assert missing == ["VA_ANTHROPIC_API_KEY"]

    def test_satisfied_when_keys_are_present(self) -> None:
        settings = build(
            profile=Profile.LOW_LATENCY,
            deepgram_api_key="dg",
            elevenlabs_api_key="el",
            openai_api_key="oa",
        )
        assert settings.missing_credentials() == []


class TestLatencyBudget:
    def test_stages_sum_to_the_published_target(self) -> None:
        # bench and evals assert against this object, so the number cannot drift
        # away from the prose. It was 800ms, built on a 250ms endpoint that
        # measurement showed cut the caller off on a third of turns; the
        # endpoint line moved to 450ms and the total moved with it rather than
        # the system being reported against a target it could not honestly meet.
        assert LatencyBudget().total_ms == 1000

    def test_the_budget_describes_the_system_that_actually_runs(self) -> None:
        """The endpoint line and the endpointer must not drift apart.

        They are two statements of one fact. If the budget claims 450ms of
        endpointing while the pipeline waits 250, every published figure is
        measured against a system nobody is running.
        """
        assert LatencyBudget().endpoint_detection == Settings().stop_hang_ms

    def test_p95_is_looser_than_p50(self) -> None:
        budget = LatencyBudget()
        assert budget.p95_ms > budget.total_ms


class TestDerivedValues:
    def test_barge_in_is_on_by_default_for_the_browser_demo(self) -> None:
        settings = build()
        assert settings.barge_in is True
        assert settings.barge_in_min_ms == 500

    def test_millisecond_settings_expose_seconds_for_asyncio(self) -> None:
        settings = build(stop_hang_ms=250, post_tts_guard_ms=300)
        assert settings.stop_hang_s == 0.25
        assert settings.post_tts_guard_s == 0.3


class TestPiperDiscovery:
    """The quick start has to be true end to end.

    ``scripts/fetch_models.py`` always closed by saying "the defaults find these
    automatically" and nothing did — the only working setup was an absolute path
    someone had exported by hand. A fresh clone followed the documented steps and
    then failed to start.
    """

    @staticmethod
    def _install(root: Path, *names: str) -> None:
        (root / "models" / "piper").mkdir(parents=True)
        (root / "models" / "piper" / "piper.exe").write_bytes(b"")
        voices = root / "models" / "piper" / "voices"
        voices.mkdir()
        for name in names:
            (voices / f"{name}.onnx").write_bytes(b"")

    def test_finds_the_binary_and_the_chosen_voice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install(tmp_path, "en_GB-alba-medium", "en_US-ryan-medium")
        monkeypatch.chdir(tmp_path)

        settings = build()
        assert settings.piper_bin is not None
        assert settings.piper_voice is not None
        assert settings.piper_voice.stem == "en_GB-alba-medium"

    def test_any_voice_beats_refusing_to_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mismatched accent is a smaller problem than a server that will not boot."""
        self._install(tmp_path, "en_US-ryan-medium")
        monkeypatch.chdir(tmp_path)

        settings = build()
        assert settings.piper_voice is not None
        assert settings.piper_voice.stem == "en_US-ryan-medium"

    def test_an_explicit_setting_still_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install(tmp_path, "en_GB-alba-medium", "en_US-ryan-medium")
        monkeypatch.chdir(tmp_path)
        chosen = tmp_path / "models" / "piper" / "voices" / "en_US-ryan-medium.onnx"

        assert build(piper_voice=chosen).piper_voice == chosen


class TestVoicePerLanguage:
    @staticmethod
    def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
        TestPiperDiscovery._install(
            tmp_path, "en_GB-alba-medium", "en_GB-alan-medium", "de_DE-thorsten-medium"
        )
        monkeypatch.chdir(tmp_path)
        return build()

    def test_the_configured_voice_wins_for_its_own_language(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise the lookup silently overrides a choice made on measurement.

        Sorted alphabetically, "alan" comes before "alba" — so a naive lookup
        returned a voice nobody picked.
        """
        settings = self._settings(tmp_path, monkeypatch)
        voice = settings.voice_for("en")
        assert voice is not None
        assert voice.stem == "en_GB-alba-medium"

    def test_switches_voice_for_a_detected_language(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = self._settings(tmp_path, monkeypatch)
        voice = settings.voice_for("de")
        assert voice is not None
        assert voice.stem == "de_DE-thorsten-medium"

    def test_piper_loads_installed_switch_voices_even_when_asr_is_english_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = self._settings(tmp_path, monkeypatch)
        assert settings.languages == ("en",)

        tts = PiperTTS.from_settings(settings)

        assert tts._voices["de"].stem == "de_DE-thorsten-medium"

    def test_an_uninstalled_language_falls_back_rather_than_failing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Speaking English at someone is better than silence."""
        settings = self._settings(tmp_path, monkeypatch)
        assert settings.voice_for("fr") == settings.piper_voice
