"""Per-component latency probes.

Each probe measures one stage of the pipeline in isolation, against the real
binary or service, before any of the pipeline exists. The point is to discover
in phase 1 whether the 800 ms first-audio budget is reachable on a given stack —
finding that out in phase 5, after the architecture has been shaped around an
assumption, is how projects end up with a target they quietly stop mentioning.

Probes degrade rather than fail: an absent component is reported as unavailable
so the table shows what was actually measured on this machine.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from bench.stats import Measurement, repeat, repeat_reported

#: Representative of what the agent actually says: short, spoken, one clause.
SHORT_CLAUSE = "Sure, let me check that for you."
FULL_SENTENCE = (
    "I can see three open orders on your account, and the most recent one "
    "shipped on Tuesday morning."
)


@dataclass(frozen=True)
class PiperConfig:
    binary: Path
    voice: Path


def find_piper() -> PiperConfig | None:
    """Locate a Piper binary and voice.

    Checks the environment first, then the conventional layout that
    ``scripts/fetch_models.py`` produces.
    """
    env_bin = os.environ.get("VA_PIPER_BIN")
    env_voice = os.environ.get("VA_PIPER_VOICE")
    candidates: list[Path] = []
    if env_bin:
        candidates.append(Path(env_bin))
    for root in (Path.cwd() / "models" / "piper", Path.cwd() / "piper"):
        candidates.extend([root / "piper.exe", root / "piper"])
    binary = next((p for p in candidates if p.is_file()), None)
    if binary is None:
        found = shutil.which("piper")
        binary = Path(found) if found else None
    if binary is None:
        return None

    if env_voice and Path(env_voice).is_file():
        return PiperConfig(binary=binary, voice=Path(env_voice))

    voices_dir = binary.parent / "voices"
    if not voices_dir.is_dir():
        return None
    # Prefer a medium model: the high-quality ones are markedly slower, which
    # matters more than the quality difference at telephone bandwidth.
    onnx = sorted(voices_dir.glob("*.onnx"))
    if not onnx:
        return None
    preferred = next((p for p in onnx if "medium" in p.name), onnx[0])
    return PiperConfig(binary=binary, voice=preferred)


def synthesize(config: PiperConfig, text: str) -> bytes:
    """One Piper synthesis, returning WAV bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.wav"
        proc = subprocess.run(
            [
                str(config.binary),
                "-m",
                str(config.voice),
                "-f",
                str(out),
                "--length_scale",
                "0.95",
            ],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"piper exited {proc.returncode}: {proc.stderr.decode('utf-8', 'ignore')[:200]}"
            )
        return out.read_bytes()


def probe_tts(runs: int) -> list[Measurement]:
    """Time Piper synthesis for a short clause and a full sentence.

    Piper is not a streaming engine: it emits the whole utterance at once, so
    time-to-first-audio equals total synthesis time. That is the single most
    important finding for pipeline design on the local stack, because it means
    the only lever available is *synthesising less text at a time* — which is
    precisely why the pipeline triggers TTS on a clause rather than a sentence.
    """
    config = find_piper()
    if config is None:
        note = "piper binary or voice not found (run scripts/fetch_models.py)"
        return [
            Measurement("TTS (Piper)", "short clause", unavailable=note),
            Measurement("TTS (Piper)", "full sentence", unavailable=note),
        ]

    results = []
    for label, text in (("short clause", SHORT_CLAUSE), ("full sentence", FULL_SENTENCE)):
        m = Measurement("TTS (Piper)", label)
        results.append(repeat(m, lambda t=text: synthesize(config, t), runs=runs))
    return results


class ResidentPiper:
    """A Piper process kept alive across requests, streaming raw PCM.

    The wrapper this replaces spawned ``piper.exe`` per request and waited for a
    complete WAV file, paying process start and ONNX graph load on every clause.
    Piper supports ``--json-input`` (stay resident, read requests as JSON lines)
    and ``--output_raw`` (emit PCM as it is produced), so both costs are
    avoidable — this probe measures how much that is worth.

    A reader thread drains stdout continuously. With raw output there is no
    delimiter between utterances, so timing is taken as "first new audio after
    the request was written", which is exactly the quantity the caller perceives.
    """

    def __init__(self, config: PiperConfig) -> None:
        self._proc = subprocess.Popen(
            [
                str(config.binary),
                "-m",
                str(config.voice),
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
        self._chunks: queue.Queue[float] = queue.Queue()
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        assert self._proc.stdout is not None
        while self._proc.stdout.read(4096):
            self._chunks.put(time.perf_counter())

    def _quiesce(self, idle_ms: int = 400) -> None:
        """Block until no audio has arrived for ``idle_ms``, then discard it all.

        Without this the previous utterance is still streaming when the next
        request is written, and the timer stops on *its* tail — which reads as a
        first-chunk latency of zero. Draining with ``empty()`` is not enough
        because the reader thread keeps appending after the drain loop exits.
        """
        while True:
            try:
                self._chunks.get(timeout=idle_ms / 1000.0)
            except queue.Empty:
                return

    def first_chunk_ms(self, text: str) -> float:
        """Milliseconds from request to the first byte of audio."""
        assert self._proc.stdin is not None
        self._quiesce()

        start = time.perf_counter()
        self._proc.stdin.write((json.dumps({"text": text}) + "\n").encode())
        self._proc.stdin.flush()
        try:
            arrived = self._chunks.get(timeout=30)
        except queue.Empty as exc:
            raise RuntimeError("resident piper produced no audio") from exc
        return (arrived - start) * 1000.0

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.terminate()
        self._proc.wait(timeout=10)


def probe_tts_resident(runs: int) -> list[Measurement]:
    """Time-to-first-audio from a warm, resident Piper process.

    The difference between this and :func:`probe_tts` is entirely process start
    and model load — a cost paid per clause by the old design and never by this
    one. It is the phase-1 finding that decides how the TTS adapter is built.
    """
    config = find_piper()
    if config is None:
        return [
            Measurement(
                "TTS (Piper, resident)",
                "first audio",
                unavailable="piper binary or voice not found",
            )
        ]

    piper = ResidentPiper(config)
    try:
        results = []
        for label, text in (
            ("first audio — short clause", SHORT_CLAUSE),
            ("first audio — full sentence", FULL_SENTENCE),
        ):
            m = Measurement("TTS (Piper, resident)", label)
            results.append(repeat_reported(m, lambda t=text: piper.first_chunk_ms(t), runs=runs))
        return results
    finally:
        piper.close()


def _ollama_tags(url: str) -> list[str]:
    with urllib.request.urlopen(f"{url}/api/tags", timeout=4) as response:
        payload = json.loads(response.read())
    return [m["name"] for m in payload.get("models", []) if m.get("name")]


def ollama_first_token_ms(url: str, model: str, prompt: str) -> float:
    """Time to the first streamed token, which is what the caller waits on.

    Total generation time is nearly irrelevant to perceived latency: the
    pipeline starts speaking as soon as it has a clause, so what matters is how
    quickly the first tokens arrive.
    """
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "5m",
            "options": {"temperature": 0.4, "num_predict": 60},
        }
    ).encode()
    request = urllib.request.Request(
        f"{url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        for line in response:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("response"):
                return (time.perf_counter() - start) * 1000.0
    raise RuntimeError("stream produced no tokens")


def probe_llm(runs: int, url: str) -> list[Measurement]:
    """Time-to-first-token for every installed Ollama model."""
    try:
        models = _ollama_tags(url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return [
            Measurement(
                "LLM (Ollama)",
                "time to first token",
                unavailable=f"ollama unreachable at {url}: {exc}",
            )
        ]
    if not models:
        return [
            Measurement("LLM (Ollama)", "time to first token", unavailable="no models installed")
        ]

    prompt = (
        "You are a concise phone agent. Reply in one short sentence.\n"
        "User: Hi, can you check my most recent order?\nAssistant:"
    )
    results = []
    for model in models:
        m = Measurement("LLM (Ollama)", f"first token — {model}")
        results.append(
            repeat(m, lambda mo=model: ollama_first_token_ms(url, mo, prompt), runs=runs)
        )
    return results


def probe_asr(runs: int, device: str) -> list[Measurement]:
    """Time faster-whisper on utterances of realistic length.

    Batch ASR runs *after* the caller stops speaking, so its full duration lands
    in the first-audio budget. A streaming engine overlaps with the speech and
    only pays a finalisation cost — that difference is the whole argument for
    the cloud profile, and this probe is what makes it concrete rather than
    asserted.
    """
    # Must run before faster-whisper imports ctranslate2, which resolves its CUDA
    # dependencies at load time. Without it, GPU construction stalls silently.
    from voice_agent.providers.cuda import register_cuda_libraries

    register_cuda_libraries()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return [
            Measurement(
                "ASR (faster-whisper)",
                "batch transcribe",
                unavailable="faster-whisper not installed (uv sync --extra local)",
            )
        ]

    config = find_piper()
    if config is None:
        return [
            Measurement(
                "ASR (faster-whisper)",
                "batch transcribe",
                unavailable="needs piper to synthesise test audio",
            )
        ]

    from voice_agent.audio import wav

    model_name = os.environ.get("VA_WHISPER_MODEL", "small.en")
    try:
        model = WhisperModel(model_name, device=device, compute_type="auto")
    except Exception as exc:
        return [
            Measurement(
                "ASR (faster-whisper)",
                f"batch transcribe ({device})",
                unavailable=f"{type(exc).__name__}: {exc}",
            )
        ]

    def transcribe(audio: object) -> str:
        segments, _ = model.transcribe(audio, language="en", beam_size=1)
        return " ".join(s.text for s in segments)

    results = []
    for label, text in (("~2s utterance", SHORT_CLAUSE), ("~6s utterance", FULL_SENTENCE)):
        pcm = wav.decode_wav(synthesize(config, text))
        audio = wav.to_float32(pcm)
        seconds = len(pcm) / (16_000 * 2)
        m = Measurement("ASR (faster-whisper)", f"{label} ({seconds:.1f}s, {model_name}, {device})")
        results.append(repeat(m, lambda a=audio: transcribe(a), runs=runs))
    return results
