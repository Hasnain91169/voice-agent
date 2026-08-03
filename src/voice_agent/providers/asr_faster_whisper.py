"""faster-whisper ASR.

Batch transcription: the whole utterance is handed over once the VAD has decided
the caller stopped speaking, so this stage's full duration lands inside the
first-audio budget. Measured at 90ms on an RTX 5070 and 1504ms on CPU for the
same 2.5s clip — which is why device selection decides whether the local stack
meets the latency target at all.

The model is blocking C++ under the hood, so every call runs in a worker thread.
Doing it inline would stall the event loop for the duration of the transcribe,
which on CPU means stalling the RX pump — and therefore the barge-in detector —
for a second and a half.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

import numpy as np

from voice_agent.audio import wav
from voice_agent.config import SAMPLE_RATE, Settings
from voice_agent.providers.base import Health, Transcript
from voice_agent.providers.cuda import register_cuda_libraries

log = logging.getLogger(__name__)

#: Segments whose speech probability falls below this are treated as silence
#: rather than transcribed noise.
_NO_SPEECH_CEILING = 0.6


class FasterWhisperASR:
    """Local Whisper transcription via CTranslate2."""

    name = "faster_whisper"

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        compute_type: str = "auto",
        cpu_threads: int = 8,
        beam_size: int = 1,
        language: str | None = "en",
    ) -> None:
        self._model_name = model_name
        self._requested_device = device
        self._requested_compute = compute_type
        self._cpu_threads = cpu_threads
        self._beam_size = beam_size
        #: ``None`` means detect per utterance. Pinning it is faster and more
        #: accurate when only one language is expected.
        self._language = language
        self._model: Any = None
        self._lock = asyncio.Lock()
        self.device = "cpu"
        self.compute_type = "int8"

    @classmethod
    def from_settings(cls, settings: Settings) -> FasterWhisperASR:
        return cls(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            cpu_threads=settings.whisper_cpu_threads,
            language=None if settings.multilingual else settings.languages[0],
        )

    def _load(self) -> Any:
        """Construct the model. Blocking; always called in a worker thread."""
        # Must precede the ctranslate2 import: it resolves cuBLAS lazily at the
        # first inference through a plain LoadLibrary that consults PATH.
        register_cuda_libraries()
        from faster_whisper import WhisperModel

        device, compute = self._resolve_device()
        try:
            model = WhisperModel(
                self._model_name,
                device=device,
                compute_type=compute,
                cpu_threads=self._cpu_threads if device == "cpu" else 0,
            )
        except Exception as exc:
            if device == "cpu":
                raise
            log.warning("CUDA load failed (%s); falling back to CPU", exc)
            device, compute = "cpu", "int8"
            model = WhisperModel(
                self._model_name,
                device=device,
                compute_type=compute,
                cpu_threads=self._cpu_threads,
            )

        self.device, self.compute_type = device, compute
        log.info(
            "faster-whisper loaded model=%s device=%s compute=%s",
            self._model_name,
            device,
            compute,
        )
        return model

    def _resolve_device(self) -> tuple[str, str]:
        if self._requested_device in ("cpu", "cuda"):
            device = self._requested_device
        else:
            device = "cuda" if _cuda_available() else "cpu"
        if self._requested_compute != "auto":
            return device, self._requested_compute
        return device, ("float16" if device == "cuda" else "int8")

    async def _ensure_loaded(self) -> Any:
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load)
            return self._model

    async def transcribe(self, pcm: bytes) -> Transcript:
        if not pcm:
            return Transcript(text="", confidence=0.0)
        model = await self._ensure_loaded()
        return await asyncio.to_thread(self._transcribe_sync, model, pcm)

    def _transcribe_sync(self, model: Any, pcm: bytes) -> Transcript:
        audio = wav.to_float32(pcm)
        segments, info = model.transcribe(
            audio,
            # None means detect. Pinned to a language, Whisper transcribes
            # German as though it were English and returns something fluent and
            # wrong, which is far harder to notice than a failure.
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=False,  # the pipeline's own VAD already segmented this
        )

        parts: list[str] = []
        confidences: list[float] = []
        for segment in segments:
            if getattr(segment, "no_speech_prob", 0.0) > _NO_SPEECH_CEILING:
                continue
            text = segment.text.strip()
            if not text:
                continue
            parts.append(text)
            confidences.append(_normalise_confidence(segment))

        return Transcript(
            text=" ".join(parts).strip(),
            # The weakest segment governs: one badly-heard clause is enough to
            # make the whole utterance worth re-prompting.
            confidence=min(confidences) if confidences else 0.0,
            language=getattr(info, "language", None),
        )

    async def warmup(self) -> None:
        """Load the model and run one inference.

        Both halves matter. The first CUDA inference was measured at 9.26s
        against 50ms warm, because kernels are compiled for the device on first
        use — loading the model alone does not pay that cost.
        """
        try:
            await self.transcribe(wav.silence(500))
            log.info("faster-whisper warmed (%s/%s)", self.device, self.compute_type)
        except Exception as exc:
            log.warning("faster-whisper warmup failed: %s", exc)

    async def health(self) -> Health:
        start = time.perf_counter()
        try:
            await self.transcribe(wav.silence(200))
            return Health(
                ok=True,
                detail=f"{self._model_name} on {self.device}/{self.compute_type}",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        except Exception as exc:
            return Health(ok=False, detail=f"{type(exc).__name__}: {exc}")

    async def aclose(self) -> None:
        self._model = None


def _normalise_confidence(segment: Any) -> float:
    """Map a Whisper segment's average log-probability onto 0..1.

    Providers report confidence in incompatible units; normalising here means
    the pipeline's low-confidence branch does not need to know which ASR is
    behind it.
    """
    avg_logprob = getattr(segment, "avg_logprob", None)
    if avg_logprob is None:
        return 1.0
    return float(np.clip(math.exp(avg_logprob), 0.0, 1.0))


def _cuda_available() -> bool:
    try:
        register_cuda_libraries()
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:
        return False


__all__ = ["SAMPLE_RATE", "FasterWhisperASR"]
