"""Piper TTS, kept resident.

The measurement that shaped this file: spawning ``piper`` per clause reaches
first audio in 466ms; the same binary kept alive reaches it in **123ms**. The
~340ms difference is process start plus ONNX graph load, and the previous
implementation paid it on every clause of every turn.

Two Piper flags make that possible. ``--json-input`` keeps the process reading
requests as JSON lines instead of exiting after one, and ``--output_raw`` emits
PCM as it is produced rather than writing a complete WAV file at the end.

The cost is that raw output has no delimiter between utterances, so the end of a
clause is detected by the stream going idle. That costs nothing on the latency
that matters — audio is already flowing to the caller by then — it only delays
when the *next* clause starts synthesising. Piper runs far faster than realtime,
so serialising clauses still keeps the playback queue fed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from voice_agent.agent import locale
from voice_agent.audio import wav
from voice_agent.audio.framing import Frame  # noqa: F401  (documents the shared format)
from voice_agent.config import BYTES_PER_FRAME, SAMPLE_RATE, Settings
from voice_agent.providers.base import Health

log = logging.getLogger(__name__)

#: Read size from Piper's stdout. Small enough that first audio is handed on
#: promptly, large enough not to syscall per sample.
_READ_BYTES = 4_096

#: Silence on stdout that marks the end of an utterance. Piper produces audio
#: continuously while synthesising, so a gap this long means it has finished.
_IDLE_MS = 120

#: How long to wait for the *first* audio of a clause before giving up.
_FIRST_AUDIO_TIMEOUT_S = 20.0


class PiperTTS:
    """A resident Piper process exposed as a streaming TTS provider."""

    name = "piper"

    def __init__(self, binary: Path, voice: Path, voices: dict[str, Path] | None = None) -> None:
        self._binary = binary
        self._voice = voice
        #: One voice per language, resolved at startup. Switching means
        #: restarting the process — Piper loads exactly one model — so this is
        #: a per-turn cost of a few hundred milliseconds and only when the
        #: caller actually changes language, not on every turn.
        self._voices = voices or {}
        self._process: asyncio.subprocess.Process | None = None
        #: Piper synthesises one request at a time; the lock keeps concurrent
        #: callers from interleaving their audio on the shared stdout.
        self._lock = asyncio.Lock()
        self._native_rate = _read_voice_rate(voice)
        #: True while a clause's audio may still be unread — set before reading
        #: and cleared only on normal completion, so a clause abandoned by
        #: barge-in leaves it set and the next call knows to drain first.
        self._dirty = False
        #: Processes retired by a voice switch, kept only until they are
        #: reaped. Dropping the reference without waiting leaves asyncio to
        #: collect the transport at interpreter shutdown, which surfaces as
        #: 'Event loop is closed' tracebacks after a clean run.
        self._reapers: set[asyncio.Task[None]] = set()

    @classmethod
    def from_settings(cls, settings: Settings) -> PiperTTS:
        if settings.piper_bin is None or settings.piper_voice is None:
            raise RuntimeError(
                "Piper is not configured. Run `python scripts/fetch_models.py`, "
                "or set VA_PIPER_BIN and VA_PIPER_VOICE."
            )
        voice_languages = set(settings.languages) | set(locale.SUPPORTED)
        voices = {lang: v for lang in voice_languages if (v := settings.voice_for(lang))}
        return cls(binary=settings.piper_bin, voice=settings.piper_voice, voices=voices)

    def use_language(self, language: str) -> None:
        """Switch to this language's voice for subsequent synthesis.

        A no-op when the voice is already right, which is the common case: a
        caller who speaks German twice in a row pays the restart once.
        """
        voice = self._voices.get(language)
        if voice is None:
            # Say so. Silently keeping the current voice means German comes out
            # in an English one, which sounds like broken language detection
            # rather than a configuration that never enabled German — a trap
            # that cost an afternoon of blaming the recogniser.
            log.warning(
                "no voice configured for %r; staying on %s. Add it to VA_LANGUAGES.",
                language,
                self._voice.name,
            )
            return
        if voice == self._voice:
            return
        log.info("switching voice %s -> %s", self._voice.name, voice.name)
        self._voice = voice
        self._native_rate = _read_voice_rate(voice)
        # Piper holds one model. The process is torn down here and restarted
        # lazily by _ensure_started on the next synthesis, so the cost lands
        # before the first clause rather than mid-sentence.
        self._retire_process()

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process

        if not self._binary.is_file():
            raise FileNotFoundError(f"piper binary not found: {self._binary}")
        if not self._voice.is_file():
            raise FileNotFoundError(f"piper voice not found: {self._voice}")

        self._process = await asyncio.create_subprocess_exec(
            str(self._binary),
            "-m",
            str(self._voice),
            "--output_raw",
            "--json-input",
            "--length_scale",
            "0.95",
            # Piper appends 200ms of silence to every utterance it synthesises.
            # That is right for a whole sentence and wrong here, because a
            # clause is not the end of a thought — the pipeline splits on
            # commas, so a three-clause answer collected 600ms of dead air in
            # the middle of one spoken sentence. Measured: clause splitting
            # added 922ms of audio to a sentence that is 6.8s whole.
            "--sentence_silence",
            "0.0",
            "-q",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        log.info(
            "piper started pid=%s voice=%s native_rate=%d",
            self._process.pid,
            self._voice.name,
            self._native_rate,
        )
        return self._process

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield 16 kHz PCM for ``text`` as Piper produces it."""
        clean = text.strip()
        if not clean:
            return

        async with self._lock:
            process = await self._ensure_started()
            assert process.stdin is not None and process.stdout is not None

            # Only drain when a previous clause was abandoned mid-stream, which
            # leaves audio in the pipe that would otherwise be prepended to this
            # one and played as the wrong words. Draining unconditionally would
            # add the full idle window to *every* clause — measured at 255ms
            # first-audio against 123ms without it, most of the TTS budget spent
            # waiting for a pipe that is almost always already empty.
            if self._dirty:
                await self._drain(process)
                self._dirty = False

            request = json.dumps({"text": clean}, ensure_ascii=False) + "\n"
            process.stdin.write(request.encode("utf-8"))
            await process.stdin.drain()
            self._dirty = True

            resampler = wav.StreamingResampler(self._native_rate, SAMPLE_RATE)
            trimmer = _EdgeTrimmer()
            started = False
            deadline = time.monotonic() + _FIRST_AUDIO_TIMEOUT_S

            while True:
                timeout = _IDLE_MS / 1000.0 if started else 0.25
                try:
                    chunk = await asyncio.wait_for(
                        process.stdout.read(_READ_BYTES), timeout=timeout
                    )
                except TimeoutError:
                    if started:
                        break  # gap in the stream: the clause is finished
                    if time.monotonic() > deadline:
                        raise RuntimeError("piper produced no audio") from None
                    continue

                if not chunk:  # process exited
                    break

                started = True
                converted = resampler.process(chunk)
                if converted:
                    for piece in trimmer.feed(converted):
                        yield piece

            # Reached only on normal completion; a cancelled generator leaves
            # _dirty set so the next caller drains the abandoned tail.
            self._dirty = False

            tail = resampler.flush()
            if tail:
                for piece in trimmer.feed(tail):
                    yield piece
            for piece in trimmer.finish():
                yield piece

    async def _drain(self, process: asyncio.subprocess.Process) -> None:
        """Discard any audio left over from an abandoned clause."""
        assert process.stdout is not None
        while True:
            try:
                chunk = await asyncio.wait_for(
                    process.stdout.read(_READ_BYTES), timeout=_IDLE_MS / 1000.0
                )
            except TimeoutError:
                return
            if not chunk:
                return

    async def warmup(self) -> None:
        """Start the process and synthesise once, before any caller waits on it."""
        try:
            async for _ in self.synthesize("Ready."):
                pass
            log.info("piper warmed")
        except Exception as exc:
            log.warning("piper warmup failed: %s", exc)

    async def health(self) -> Health:
        start = time.perf_counter()
        try:
            total = 0
            async for chunk in self.synthesize("ok"):
                total += len(chunk)
            elapsed = (time.perf_counter() - start) * 1000.0
            if total == 0:
                return Health(ok=False, detail="produced no audio", latency_ms=elapsed)
            return Health(ok=True, detail=f"{total} bytes", latency_ms=elapsed)
        except Exception as exc:
            return Health(ok=False, detail=f"{type(exc).__name__}: {exc}")

    def _retire_process(self) -> None:
        """Drop the running Piper so the next synthesis starts a new one.

        Synchronous on purpose: a voice switch happens between turns, from
        ordinary code rather than from an await point, and making it async
        would push the restart into the middle of the first clause. The process
        is signalled and left for the loop to reap — nothing reads its output
        after this, because ``_process`` is already None.
        """
        process = self._process
        self._process = None
        self._dirty = False
        if process is None or process.returncode is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        with contextlib.suppress(ProcessLookupError):
            process.terminate()

        # Wait for it somewhere else. Not awaiting at all leaves the transport
        # for __del__ to close after the loop has gone, which prints a
        # traceback on an otherwise clean shutdown.
        with contextlib.suppress(RuntimeError):  # no running loop: nothing to schedule
            task = asyncio.get_running_loop().create_task(_reap(process))
            self._reapers.add(task)
            task.add_done_callback(self._reapers.discard)

    async def aclose(self) -> None:
        if self._reapers:
            await asyncio.gather(*tuple(self._reapers), return_exceptions=True)
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=5.0)


#: Below this RMS a 20ms frame is silence rather than speech. Well under the
#: quietest voiced frame and well over the dither in a synthesised signal.
_SILENCE_RMS = 180

#: Left on the end of every clause. Running clauses together with no gap at all
#: sounds hurried and swallows the comma; this is roughly the pause a person
#: leaves mid-sentence, and it is the same every time rather than whatever the
#: model happened to generate.
_TAIL_MS = 60


class _EdgeTrimmer:
    """Removes the silence Piper puts either side of a clause.

    Leading silence is pure latency — the caller waits and hears nothing — so
    it goes entirely. Trailing silence is held back and replaced with a fixed
    short tail, because the pipeline speaks a clause at a time and the model's
    own trailing pause lands in the middle of a sentence.

    Streaming-safe: audio is emitted as it arrives once speech has started,
    with only a rolling window held back so the end can be trimmed without
    waiting for the whole clause.
    """

    __slots__ = ("_hold", "_started")

    #: Enough to cover any trailing pause without delaying playback.
    _WINDOW = SAMPLE_RATE * 2 * 500 // 1000  # 500ms of 16-bit mono

    def __init__(self) -> None:
        self._started = False
        self._hold = bytearray()

    def feed(self, pcm: bytes) -> list[bytes]:
        if not self._started:
            offset = _first_voiced(pcm)
            if offset is None:
                return []  # still silent: drop it, the caller is waiting
            self._started = True
            pcm = pcm[offset:]

        self._hold += pcm
        if len(self._hold) <= self._WINDOW:
            return []
        cut = len(self._hold) - self._WINDOW
        out = bytes(self._hold[:cut])
        del self._hold[:cut]
        return [out]

    def finish(self) -> list[bytes]:
        """Emit what is held, trimmed to a fixed tail."""
        if not self._hold:
            return []
        held = bytes(self._hold)
        self._hold.clear()
        end = _last_voiced(held)
        if end is None:
            return []
        tail = SAMPLE_RATE * 2 * _TAIL_MS // 1000
        return [held[: min(len(held), end + tail)]]


def _frames(pcm: bytes) -> range:
    return range(0, len(pcm) - BYTES_PER_FRAME + 1, BYTES_PER_FRAME)


def _first_voiced(pcm: bytes) -> int | None:
    for offset in _frames(pcm):
        if wav.rms(pcm[offset : offset + BYTES_PER_FRAME]) >= _SILENCE_RMS:
            return offset
    return None


def _last_voiced(pcm: bytes) -> int | None:
    for offset in reversed(_frames(pcm)):
        if wav.rms(pcm[offset : offset + BYTES_PER_FRAME]) >= _SILENCE_RMS:
            return offset + BYTES_PER_FRAME
    return None


async def _reap(process: asyncio.subprocess.Process) -> None:
    """Collect a terminated Piper so its transport closes on our terms."""
    with contextlib.suppress(TimeoutError, ProcessLookupError):
        await asyncio.wait_for(process.wait(), timeout=5.0)


def _read_voice_rate(voice: Path) -> int:
    """Read the voice's native sample rate from its sidecar config.

    Raw output carries no WAV header, so the rate has to come from the model
    config. Guessing 22050 works for most Piper voices and produces
    chipmunk-or-drawl audio for the rest.
    """
    config_path = voice.with_suffix(voice.suffix + ".json")
    try:
        config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        rate = int(config["audio"]["sample_rate"])
    except (OSError, KeyError, ValueError, TypeError):
        log.warning("could not read sample rate from %s; assuming 22050", config_path)
        return 22_050
    return rate
