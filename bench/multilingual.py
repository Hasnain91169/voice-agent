"""What detecting the language costs, and whether it works.

    uv run python -m bench.multilingual

``small.en`` is faster and more accurate at English than the multilingual
``small`` of the same size, and it cannot detect a language — asked for German
it returns fluent English nonsense, which is the failure mode hardest to notice.
Supporting German means giving that up. This measures what it costs.

The audio is synthesised by Piper and fed straight back through Whisper, which
makes the loop self-contained and also proves the two halves agree: if the
recogniser cannot identify the language of speech this system produced itself,
it will not manage a rep in a van.

That is a friendlier test than a real call — clean audio, no packet loss, no
background noise — so read the transcription accuracy as a ceiling rather than
an expectation. The latency figures are the point.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from voice_agent.audio import wav
from voice_agent.config import Settings
from voice_agent.providers.tts_piper import PiperTTS

#: Same content in both languages, and both awkward on purpose: a company name
#: that is not a word, a number, and a date.
LINES = {
    "en": "Marchwood Timber are eighty-five days without an order, since the seventeenth of May.",
    "de": (
        "Marchwood Timber hat seit dem siebzehnten Mai nicht bestellt, "
        "das sind fünfundachtzig Tage."
    ),
}


@dataclass
class Row:
    model: str
    spoken: str
    pinned: str | None
    detected: str | None = None
    text: str = ""
    ms: list[float] = field(default_factory=list)
    error: str = ""

    @property
    def p50(self) -> float:
        return statistics.median(self.ms) if self.ms else 0.0


async def synthesise(settings: Settings) -> dict[str, bytes]:
    """One utterance per language, each in its own voice."""
    tts = PiperTTS.from_settings(settings)
    audio: dict[str, bytes] = {}
    try:
        for language, text in LINES.items():
            tts.use_language(language)
            audio[language] = b"".join([chunk async for chunk in tts.synthesize(text)])
    finally:
        await tts.aclose()
    return audio


def measure(model_name: str, pcm: bytes, pinned: str | None, runs: int) -> Row:
    row = Row(model=model_name, spoken="", pinned=pinned)
    # Must run before faster-whisper imports ctranslate2, which resolves its
    # CUDA dependencies at load time. Without it the model constructs fine and
    # then dies on the first inference with a missing cublas DLL.
    from voice_agent.providers.cuda import register_cuda_libraries

    register_cuda_libraries()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        row.error = "faster-whisper not installed (uv sync --extra local)"
        return row

    settings = Settings()
    try:
        model = WhisperModel(
            model_name,
            device="cuda" if settings.whisper_device != "cpu" else "cpu",
            compute_type="auto",
        )
    except Exception as exc:  # a missing model must not stop the sweep
        row.error = f"{type(exc).__name__}: {exc}"
        return row

    audio = wav.to_float32(pcm)
    # One discarded run: the first inference compiles kernels, which is 9s on
    # CUDA and not a cost any caller ever sees on a warm server.
    model.transcribe(audio, language=pinned, beam_size=1, vad_filter=False)

    for _ in range(runs):
        started = time.perf_counter()
        segments, info = model.transcribe(audio, language=pinned, beam_size=1, vad_filter=False)
        text = " ".join(segment.text.strip() for segment in segments)
        row.ms.append((time.perf_counter() - started) * 1000.0)
        row.detected = getattr(info, "language", None)
        row.text = text.strip()
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.multilingual", description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args(argv)

    # Explicitly bilingual: PiperTTS only loads voices for configured
    # languages, so a default Settings would leave use_language("de") with
    # nothing to switch to and quietly synthesise German in an English voice.
    settings = Settings(languages=("en", "de"), whisper_model="small")
    if settings.piper_voice is None:
        print("No Piper voice — run: uv run python scripts/fetch_models.py --voice all")
        return 2
    if settings.voice_for("de") == settings.voice_for("en"):
        print("No German voice installed — run: scripts/fetch_models.py --voice all")
        return 2

    audio = asyncio.run(synthesise(settings))
    for language, pcm in audio.items():
        Path("models/voice-samples").mkdir(parents=True, exist_ok=True)
        Path(f"models/voice-samples/asr-probe-{language}.wav").write_bytes(
            wav.encode_wav(pcm, rate=16_000)
        )

    print(f"# Multilingual ASR — p50 of {args.runs} runs after a warm-up\n")
    rows = [
        measure("small.en", audio["en"], "en", args.runs),
        measure("small", audio["en"], None, args.runs),
        measure("small", audio["de"], None, args.runs),
        # Pinned to English on German audio: the failure the config guard exists
        # to prevent, shown rather than described.
        measure("small.en", audio["de"], "en", args.runs),
    ]
    for row, spoken in zip(rows, ("en", "en", "de", "de"), strict=True):
        row.spoken = spoken

    print("| Model | Spoken | Asked for | Detected | p50 | Transcript |")
    print("|---|:--:|:--:|:--:|--:|---|")
    for r in rows:
        if r.error:
            print(f"| {r.model} | {r.spoken} | - | - | - | **{r.error[:40]}** |")
            continue
        asked = r.pinned or "detect"
        print(
            f"| `{r.model}` | {r.spoken} | {asked} | {r.detected or '-'} | "
            f"{r.p50:.0f}ms | {r.text[:60]} |"
        )

    english = [r for r in rows if r.spoken == "en" and not r.error]
    if len(english) == 2:
        pinned, detecting = english
        cost = detecting.p50 - pinned.p50
        print(
            f"\nDetecting costs {cost:+.0f}ms on English against pinning "
            f"({pinned.p50:.0f}ms to {detecting.p50:.0f}ms), and buys a second "
            f"language. The budget allows {Settings().budget.asr_finalise}ms."
        )
    wrong = next((r for r in rows if r.model == "small.en" and r.spoken == "de"), None)
    if wrong and not wrong.error:
        print(
            "\nThe last row is why Settings refuses to start bilingual on an "
            "English-only model: asked for English, German audio does not fail, "
            "it comes back as confident nonsense."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
