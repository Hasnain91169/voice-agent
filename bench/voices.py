"""Compare Piper voices on latency, and produce samples to judge by ear.

    uv run python scripts/fetch_models.py --voice all
    uv run python -m bench.voices

Two of the three things that matter about a voice are measurable and one is not.
Time to first audio and real-time factor decide whether a voice can hold a
conversation at all; whether it sounds like a person cannot be benchmarked, so
this writes one WAV per voice saying the same sentence and leaves that judgement
to a human with headphones.

Being explicit about that split is the point. A table alone would imply the
fastest voice is the best one, which is how you end up shipping something that
answers in 90ms and sounds like a train announcement.

The sentence is deliberately awkward: a company name, a spoken number and a
spoken date. Those are what a synthesiser gets wrong, and they are most of what
this agent says.
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from voice_agent.audio import wav
from voice_agent.providers.tts_piper import _read_voice_rate

VOICES_DIR = Path("models") / "piper" / "voices"
SAMPLES_DIR = Path("models") / "voice-samples"

#: The first clause of each sentence below. **This is the number that matters.**
#:
#: Piper does not stream within an utterance — it synthesises the whole thing and
#: then flushes, which this bench demonstrates by accident: on a full sentence,
#: time-to-first-audio and total synthesis time come out identical to the
#: millisecond. So "first audio" for a long sentence is really "synthesis time
#: for a long sentence", and the pipeline never waits for one, because it splits
#: on clauses and synthesises each as it becomes speakable. Measuring the clause
#: is measuring what a caller actually waits for.
CLAUSES = {
    "en": "Marchwood Timber are eighty-five days without an order,",
    "de": "Marchwood Timber hat seit fünfundachtzig Tagen nicht bestellt,",
}

#: The clause ``bench/probes.py`` uses, kept here for comparability. The
#: published TTS figure of 89-123ms was measured on this, and it is shorter than
#: anything this agent actually says — no company name, no spoken number, no
#: date. Synthesis cost tracks audio length, so a real domain clause costs
#: roughly three times as much. Both are printed so the gap is visible rather
#: than argued about.
GENERIC_CLAUSE = "Sure, let me check that for you."

#: The full sentence, used for throughput rather than latency. Numbers and dates
#: in both languages, because those are the hard part and this agent says little
#: else.
SENTENCES = {
    "en": (
        "Marchwood Timber are eighty-five days without an order, "
        "and nobody has been in touch since the seventeenth of May."
    ),
    "de": (
        "Marchwood Timber hat seit fünfundachtzig Tagen nicht bestellt, "
        "und seit dem siebzehnten Mai hat sich niemand gemeldet."
    ),
}


@dataclass
class VoiceResult:
    name: str
    language: str
    rate: int
    #: Time to first audio for one domain clause — what a caller waits for.
    clause_ms: list[float] = field(default_factory=list)
    #: The same, for the shorter generic clause the published figure used.
    generic_ms: list[float] = field(default_factory=list)
    #: Time to synthesise the whole sentence — throughput, not latency.
    total_ms: list[float] = field(default_factory=list)
    audio_seconds: float = 0.0
    sample: Path | None = None
    error: str = ""

    @property
    def real_time_factor(self) -> float:
        """Synthesis time over audio produced. Below 1.0 is faster than real time."""
        if not self.total_ms or self.audio_seconds <= 0:
            return 0.0
        return statistics.median(self.total_ms) / 1000.0 / self.audio_seconds


class _Capture:
    """A resident Piper that keeps the audio as well as the timings.

    Deliberately separate from ``bench.probes.ResidentPiper``, which measures
    first-chunk latency and throws the audio away. Here the bytes are the
    deliverable — they become the sample you listen to — and the total
    synthesis time needs the end of the stream, not just its start.
    """

    def __init__(self, binary: Path, voice: Path) -> None:
        self._proc = subprocess.Popen(
            [
                str(binary),
                "-m",
                str(voice),
                "--output_raw",
                "--json-input",
                "--length_scale",
                "0.95",
                "-q",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._events: queue.Queue[tuple[float, bytes]] = queue.Queue()
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        assert self._proc.stdout is not None
        while chunk := self._proc.stdout.read(4096):
            self._events.put((time.perf_counter(), chunk))

    def _quiesce(self, idle_ms: int = 400) -> None:
        """Discard anything still arriving from the previous utterance.

        Raw output has no delimiter between utterances, so without this the
        timer stops on the *tail* of the last one and reports a first-chunk
        latency of zero.
        """
        while True:
            try:
                self._events.get(timeout=idle_ms / 1000.0)
            except queue.Empty:
                return

    def say(self, text: str) -> tuple[float, float, bytes]:
        """Return (ms to first audio, ms to last audio, the PCM)."""
        assert self._proc.stdin is not None
        self._quiesce()

        start = time.perf_counter()
        self._proc.stdin.write((json.dumps({"text": text}) + "\n").encode())
        self._proc.stdin.flush()

        first = self._events.get(timeout=60)
        chunks = [first[1]]
        last_at = first[0]
        # The end of an utterance is silence, not a marker.
        while True:
            try:
                at, chunk = self._events.get(timeout=0.4)
            except queue.Empty:
                break
            last_at = at
            chunks.append(chunk)

        return (
            (first[0] - start) * 1000.0,
            (last_at - start) * 1000.0,
            b"".join(chunks),
        )

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.terminate()
        self._proc.wait(timeout=10)


def find_binary() -> Path | None:
    for candidate in (
        Path("models") / "piper" / "piper.exe",
        Path("models") / "piper" / "piper",
    ):
        if candidate.is_file():
            return candidate
    return None


def measure(binary: Path, voice: Path, runs: int) -> VoiceResult:
    name = voice.stem
    language = name.split("_", 1)[0]
    clause = CLAUSES.get(language, CLAUSES["en"])
    sentence = SENTENCES.get(language, SENTENCES["en"])
    result = VoiceResult(name=name, language=language, rate=_read_voice_rate(voice))

    try:
        piper = _Capture(binary, voice)
    except OSError as exc:
        result.error = str(exc)
        return result

    try:
        # One discarded run: the first call pays the ONNX graph load, which is
        # not a cost the caller ever sees on a warm server.
        piper.say(clause)
        pcm = b""
        for _ in range(runs):
            clause_ms, _, _ = piper.say(clause)
            result.clause_ms.append(clause_ms)
            generic_ms, _, _ = piper.say(GENERIC_CLAUSE)
            result.generic_ms.append(generic_ms)
            _, total_ms, pcm = piper.say(sentence)
            result.total_ms.append(total_ms)
        result.audio_seconds = len(pcm) / (result.rate * 2)

        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        sample = SAMPLES_DIR / f"{name}.wav"
        sample.write_bytes(wav.encode_wav(pcm, rate=result.rate))
        result.sample = sample
    except Exception as exc:  # a broken voice must not stop the sweep
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        piper.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.voices", description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args(argv)

    binary = find_binary()
    if binary is None:
        print("No Piper binary — run: uv run python scripts/fetch_models.py --voice all")
        return 2
    voices = sorted(VOICES_DIR.glob("*.onnx"))
    if not voices:
        print(f"No voices in {VOICES_DIR} — run scripts/fetch_models.py --voice all")
        return 2

    print(f"# Voices — {len(voices)} candidates, p50 of {args.runs} runs after a warm-up\n")
    results = [measure(binary, voice, args.runs) for voice in voices]

    print("| Voice | Lang | Rate | Generic clause | Domain clause | Sentence | RTF |")
    print("|---|:--:|--:|--:|--:|--:|--:|")
    for r in results:
        if r.error:
            print(f"| {r.name} | {r.language} | - | **{r.error[:40]}** | | | |")
            continue
        print(
            f"| {r.name} | {r.language} | {r.rate} Hz | "
            f"{statistics.median(r.generic_ms):.0f}ms | "
            f"**{statistics.median(r.clause_ms):.0f}ms** | "
            f"{statistics.median(r.total_ms):.0f}ms | {r.real_time_factor:.2f} |"
        )

    working = [r for r in results if not r.error]
    if working:
        budget = 150  # LatencyBudget.tts_first_chunk
        generic = [r for r in working if statistics.median(r.generic_ms) <= budget]
        domain = [r for r in working if statistics.median(r.clause_ms) <= budget]
        print(
            f"\nAgainst the {budget}ms first-chunk budget: {len(generic)} of "
            f"{len(working)} voices make it on the generic clause, {len(domain)} on "
            f"a real one."
        )
        print(
            "\nTwo things this table says that the published figure does not.\n\n"
            "Piper does not stream within an utterance — it synthesises the whole "
            "thing, then flushes. On a full sentence, time-to-first-audio and total "
            "synthesis time came out identical to the millisecond. That is why the "
            "pipeline splits on clauses: a sentence would cost its entire synthesis "
            "time before a single word was audible.\n\n"
            "And synthesis cost tracks audio length, so the clause you measure "
            "decides the number you publish. A real clause here — a company name, a "
            "spoken number, a date — costs roughly three times the generic one the "
            "89-123ms figure was taken from."
        )
        print(
            "\nRanking only. A sweep leaves the machine hot and every figure above "
            "reads higher than the same voice measured alone — re-measure the one "
            "you pick with `bench --skip llm --skip asr` before publishing a number."
        )
        print(f"\nSamples in `{SAMPLES_DIR}`. Latency is measured, naturalness is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
