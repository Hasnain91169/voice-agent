"""Typed configuration for the voice pipeline.

One validated source of truth. The old gateway read ~40 bare ``os.getenv`` calls
at import time with string-literal defaults scattered across the module, which
made it impossible to see the effective configuration of a running call or to
construct a variant for testing.

Audio framing constants live here as module-level finals rather than settings:
they are protocol properties of the transports we speak (Vonage sends 16 kHz
16-bit mono; the browser is told to), not user preferences. Making them
configurable would imply a flexibility the pipeline does not have.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Audio framing (fixed by the transports, not configurable)
# ---------------------------------------------------------------------------
SAMPLE_RATE: Final = 16_000
SAMPLE_WIDTH: Final = 2  # bytes; 16-bit signed little-endian PCM
CHANNELS: Final = 1
FRAME_MS: Final = 20
SAMPLES_PER_FRAME: Final = SAMPLE_RATE * FRAME_MS // 1000  # 320
BYTES_PER_FRAME: Final = SAMPLES_PER_FRAME * SAMPLE_WIDTH * CHANNELS  # 640

#: Vonage delivers 10ms slices; we rechunk to :data:`FRAME_MS` before VAD.
INBOUND_SLICE_MS: Final = 10


class Profile(StrEnum):
    """Which stack to run.

    The two profiles exist because no single stack serves both goals: the local
    one is free and private but cannot hit the latency bar on CPU; the cloud one
    hits the bar but costs money per call and needs keys.
    """

    LOCAL_ZERO_COST = "local_zero_cost"
    LOW_LATENCY = "low_latency"


AsrProvider = Literal["faster_whisper", "deepgram"]
TtsProvider = Literal["piper", "elevenlabs"]
LlmProvider = Literal["ollama", "openai", "anthropic"]

_PROFILE_DEFAULTS: Final[dict[Profile, tuple[AsrProvider, TtsProvider, LlmProvider]]] = {
    Profile.LOCAL_ZERO_COST: ("faster_whisper", "piper", "ollama"),
    Profile.LOW_LATENCY: ("deepgram", "elevenlabs", "openai"),
}


class LatencyBudget(BaseSettings):
    """The published first-audio budget, in milliseconds.

    Measured from the caller's speech ending to the first audio byte leaving the
    server. Held here as data so ``bench/`` and ``evals/`` can assert against the
    same numbers the README publishes, instead of duplicating them in prose.
    """

    #: Raised from 250 after measuring what 250 actually cost. The original
    #: figure came from working backwards from an 800ms target, and only the
    #: latency side of the trade was ever checked. Swept in audio mode against
    #: the real endpointer:
    #:
    #:   250ms -> first audio p50 624ms, but the agent began speaking before the
    #:            caller had finished on 10 of 30 turns
    #:   450ms -> p50 938ms, overlaps down to 2 of 32
    #:   650ms -> p50 1250ms, overlaps 1 of 34
    #:
    #: Talking over the caller one turn in three is a worse call than 300ms of
    #: extra latency, and it is the failure that compounds: the caller's own
    #: continuing speech then trips barge-in and cancels the answer. The budget
    #: moves to fit the measurement rather than the measurement being reported
    #: against a target it was never going to meet.
    endpoint_detection: int = 450
    asr_finalise: int = 50
    llm_first_token: int = 200
    first_clause: int = 80
    tts_first_chunk: int = 150
    prebuffer: int = 70

    @property
    def total_ms(self) -> int:
        return (
            self.endpoint_detection
            + self.asr_finalise
            + self.llm_first_token
            + self.first_clause
            + self.tts_first_chunk
            + self.prebuffer
        )

    #: p95 is allowed to be looser than p50; a tail is acceptable, a bad median is not.
    p95_ms: int = 1200


class Settings(BaseSettings):
    """Runtime configuration, read from environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="VA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    profile: Profile = Profile.LOCAL_ZERO_COST

    # --- Provider selection. None means "take the profile default". ---
    asr_provider: AsrProvider | None = None
    tts_provider: TtsProvider | None = None
    llm_provider: LlmProvider | None = None

    # --- Local providers ---
    ollama_url: str = "http://127.0.0.1:11434"
    #: Measured at 107ms to first token, the best of the models benchmarked, and
    #: markedly better at tool-calling than the 3B alternatives — which phase 3
    #: depends on. The previous system defaulted to a model that was not even
    #: installed, so the default here is one that was actually measured.
    ollama_model: str = "qwen2.5:7b"
    ollama_keep_alive: str = "5m"
    whisper_model: str = "small.en"
    #: Languages the agent will answer in. One language keeps the English-only
    #: model, which is faster and more accurate at English than the multilingual
    #: one of the same size. Listing more switches the recogniser to
    #: auto-detection, which needs a multilingual model — ``small.en`` cannot
    #: detect anything, it can only assume.
    languages: tuple[str, ...] = ("en",)
    #: On GPU this stack meets the 800ms budget (90ms transcribe); on CPU the
    #: same model takes ~1.5s and blows it single-handedly. ``auto`` therefore
    #: decides far more than which device is used — see the README benchmarks.
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_compute_type: str = "auto"
    whisper_cpu_threads: int = 8
    #: Left unset, these are discovered under ``models/piper`` — which is where
    #: ``scripts/fetch_models.py`` installs them. The script always claimed "the
    #: defaults find these automatically" and nothing did: a fresh clone
    #: followed the documented quick start and then failed to start, because
    #: the only working configuration was an absolute path someone had set by
    #: hand months earlier.
    piper_bin: Path | None = None
    piper_voice: Path | None = None
    #: Chosen on measurement. Fastest English candidate at 271ms on a real
    #: domain clause, and British, which a UK builders' merchant wants — see
    #: ``bench/voices.py``.
    piper_default_voice: str = "en_GB-alba-medium"

    # --- Cloud providers ---
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5"
    #: Thinking depth. ``low`` keeps time-to-first-token inside the budget; the
    #: pipeline needs a speakable clause fast, not a deeply reasoned essay.
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    #: Adaptive thinking on. Turning it off is faster but can make the model
    #: write a tool call into its visible text instead of emitting a structured
    #: block — which a voice agent would then read aloud. See llm_anthropic.py.
    anthropic_thinking: bool = True
    deepgram_api_key: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"

    # --- LLM generation ---
    llm_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=120, gt=0)
    #: Play a filler line if no token has arrived by this deadline. Never dead air.
    llm_first_token_timeout_ms: int = Field(default=1_500, gt=0)
    llm_total_timeout_ms: int = Field(default=25_000, gt=0)

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8000
    session_secret: SecretStr | None = None
    #: Lifetime of the signed WebSocket handshake token.
    session_token_ttl_s: int = Field(default=120, gt=0)

    # --- Vonage telephony ---
    vonage_application_id: str | None = None
    vonage_private_key_path: Path | None = None
    public_base_url: str | None = None

    # --- Turn-taking ---
    #: Silence that ends an utterance. The single biggest latency lever we own:
    #: every millisecond here lands directly in first-audio. Lowering it clips
    #: people who pause mid-sentence, so it trades latency against interruption.
    #:
    #: That trade is now measured rather than asserted — see
    #: :class:`LatencyBudget.endpoint_detection`. At 250ms the agent started
    #: talking over the caller on a third of all turns, which is not a latency
    #: win, it is a different and worse defect wearing a good number.
    stop_hang_ms: int = Field(default=450, gt=0)
    max_utterance_s: float = Field(default=15.0, gt=0)
    #: Frames of ambient audio sampled at call start to set the noise floor.
    calibration_frames: int = Field(default=25, gt=0)
    #: Consecutive frames above threshold before we believe the caller started.
    onset_frames: int = Field(default=6, gt=0)
    #: Explicit overrides; when unset thresholds come from calibration.
    start_threshold: int | None = None
    stop_threshold: int | None = None

    # --- Absolute energy floors -------------------------------------------
    # Calibration scales thresholds to the ambient noise floor, which works on
    # a phone line where the floor is a real signal. A browser with noise
    # suppression sends a floor near zero, so the scaled threshold collapses
    # and these minimums become the only thing separating speech from silence.
    # They are expressed on the 0..32767 RMS scale: quiet room tone sits below
    # ~50, ordinary speech lands in the high hundreds to low thousands.
    #: Energy required to believe the caller has started speaking.
    vad_min_start: int = Field(default=280, gt=0)
    #: Energy below which the caller is treated as having stopped. Lower than
    #: the start threshold so a quiet syllable mid-sentence does not end a turn.
    vad_min_stop: int = Field(default=160, gt=0)
    #: Energy required to interrupt the agent mid-sentence. Deliberately well
    #: above vad_min_start: cutting the agent off by mistake is worse than
    #: taking an extra moment to notice a real interruption.
    barge_in_min_rms: int = Field(default=750, gt=0)

    # --- Echo control and barge-in ---
    barge_in: bool = False
    #: A higher bar to interrupt the agent than to start a normal turn: a false
    #: positive here cuts the agent off mid-word, which is worse than a slow start.
    barge_in_frames: int = Field(default=10, gt=0)
    #: Minimum duration of a candidate utterance before it is transcribed and
    #: allowed to make an interruption decision. This is audio duration, not
    #: wall-clock time, so a quiet syllable does not reset the requirement.
    barge_in_min_ms: int = Field(default=500, ge=0)
    #: Caller audio is dropped entirely for this long after the agent stops, to
    #: swallow the tail of our own voice returning down the line.
    post_tts_guard_ms: int = Field(default=250, ge=0)
    #: The same guard where the far end cancels echo. Browser AEC is tuned for
    #: the speaker-to-microphone path and handles headphones well, but leaks on
    #: laptop speakers at volume — so a reduced guard, not none.
    echo_cancelled_guard_ms: int = Field(default=120, ge=0)
    #: Thresholds are multiplied by this immediately after the agent speaks.
    echo_threshold_factor: float = Field(default=1.6, ge=1.0)

    # --- Playback ---
    #: Audio buffered before playback starts. Trades underrun risk against latency.
    prebuffer_ms: int = Field(default=70, ge=0)
    inter_sentence_pad_ms: int = Field(default=120, ge=0)
    #: Extra silence after playback before the mic is trusted again.
    tts_tail_guard_ms: int = Field(default=80, ge=0)

    # --- Text chunking ---
    #: Shortest fragment we will send to TTS. Below this, synthesis overhead
    #: dominates and prosody suffers.
    min_clause_chars: int = Field(default=12, gt=0)
    #: Force-flush a partial clause after this long so the caller is not left waiting.
    clause_flush_ms: int = Field(default=400, gt=0)

    # --- Agent layer ---
    #: Run the LangGraph agent (tools, checkpointed state). Falls back to a
    #: plain windowed conversation if the `agent` extra is not installed, so
    #: a missing optional dependency costs tools rather than the whole call.
    agent_enabled: bool = True
    #: The rep the assistant is working for. Tools are scoped to their book,
    #: so 'which of my accounts are slipping' has a defined answer and one
    #: rep cannot see another's customers.
    rep_name: str = "Dani Brooks"
    #: Seeded ERP/CRM database the tools read.
    db_path: Path = Path("data") / "wholesale.db"

    # --- Testing ---
    #: Fault injection for the eval harness; wraps providers to fail on demand.
    chaos: bool = False

    budget: LatencyBudget = Field(default_factory=LatencyBudget)

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> Settings:
        """Fill unset provider choices from the profile.

        Explicit settings always win, so a profile is a starting point rather
        than a straitjacket — running local ASR against a cloud LLM is a
        legitimate combination while iterating.
        """
        asr, tts, llm = _PROFILE_DEFAULTS[self.profile]
        if self.asr_provider is None:
            self.asr_provider = asr
        if self.tts_provider is None:
            self.tts_provider = tts
        if self.llm_provider is None:
            self.llm_provider = llm
        return self

    @model_validator(mode="after")
    def _require_a_multilingual_model(self) -> Settings:
        """Refuse to start bilingual on an English-only recogniser.

        ``small.en`` cannot detect a language; asked for German it returns
        confident English nonsense. Failing here means a misconfiguration
        surfaces at startup rather than as a caller being answered in the wrong
        language halfway through a call.
        """
        if len(self.languages) > 1 and self.whisper_model.endswith(".en"):
            raise ValueError(
                f"languages={self.languages} needs a multilingual Whisper model, "
                f"but whisper_model is {self.whisper_model!r}. Use 'small' rather "
                f"than 'small.en'."
            )
        return self

    @property
    def multilingual(self) -> bool:
        return len(self.languages) > 1

    @model_validator(mode="after")
    def _discover_piper(self) -> Settings:
        """Find a Piper binary and voice under ``models/piper`` if unset.

        Explicit settings win, as everywhere else. This exists so the quick
        start in the README is true end to end: fetch the models, run the
        server, talk to it — with nothing to configure by hand in between.
        """
        root = Path("models") / "piper"
        if self.piper_bin is None:
            for candidate in (root / "piper.exe", root / "piper"):
                if candidate.is_file():
                    self.piper_bin = candidate
                    break
        if self.piper_voice is None:
            preferred = root / "voices" / f"{self.piper_default_voice}.onnx"
            if preferred.is_file():
                self.piper_voice = preferred
            else:
                # Any voice beats refusing to start; a mismatched accent is a
                # smaller problem than a server that will not boot.
                self.piper_voice = next(iter(sorted((root / "voices").glob("*.onnx"))), None)
        return self

    def voice_for(self, language: str) -> Path | None:
        """The installed voice for a language, falling back to the default.

        Language is detected per utterance, so this is looked up per turn
        rather than fixed at call start.
        """
        if self.piper_voice is None:
            return None
        matches = sorted(self.piper_voice.parent.glob(f"{language}_*.onnx"))
        if not matches:
            return self.piper_voice
        # The configured voice wins for its own language. Without this the
        # lookup returned whichever installed voice sorted first, quietly
        # overriding a choice that was made on measurement.
        if self.piper_voice in matches:
            return self.piper_voice
        # Otherwise prefer the medium tier: it is the quality the rest of the
        # stack was measured against.
        medium = [v for v in matches if v.stem.endswith("-medium")]
        return (medium or matches)[0]

    @property
    def stop_hang_s(self) -> float:
        return self.stop_hang_ms / 1000.0

    @property
    def post_tts_guard_s(self) -> float:
        return self.post_tts_guard_ms / 1000.0

    def missing_credentials(self) -> list[str]:
        """Names of settings the selected providers need but do not have.

        Called at startup so a misconfigured deployment fails at the health check
        rather than halfway through the first live call.
        """
        required: list[tuple[str, object | None]] = []
        if self.asr_provider == "deepgram":
            required.append(("VA_DEEPGRAM_API_KEY", self.deepgram_api_key))
        if self.tts_provider == "elevenlabs":
            required.append(("VA_ELEVENLABS_API_KEY", self.elevenlabs_api_key))
        if self.llm_provider == "openai":
            required.append(("VA_OPENAI_API_KEY", self.openai_api_key))
        if self.llm_provider == "anthropic":
            required.append(("VA_ANTHROPIC_API_KEY", self.anthropic_api_key))
        return [name for name, value in required if value is None]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, cached.

    Tests that need a variant should construct :class:`Settings` directly rather
    than mutating the cached instance.
    """
    return Settings()
