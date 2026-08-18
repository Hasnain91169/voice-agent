"""The turn loop.

Structure of one turn:

    listen ──> transcribe ──> [ LLM ──> clauses ──> TTS ──> playback ]
                                          ▲                    │
                                          └── barge-in ────────┘

The bracketed part is a **cancel scope**. Generation, synthesis and playback run
as sibling tasks; a barge-in cancels all three through one mechanism rather than
each unwinding separately. That is what makes interruption reliable instead of
leaving a half-torn-down turn behind — the failure mode where the agent stops
speaking but keeps synthesising, then plays the tail of the abandoned sentence
over the caller's next question.

The governing rule for every failure path here is **never dead air**. Silence on
a call makes the caller start talking, which trips barge-in, which cancels the
recovery — and the call spirals. Every branch that could produce nothing instead
plays something from the pre-synthesised cache.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from voice_agent.agent import grounding, locale, prompts
from voice_agent.agent.runner import TurnSource
from voice_agent.audio import wav
from voice_agent.audio.framing import Frame
from voice_agent.audio.vad import (
    SpeechEnded,
    UtteranceDetector,
    calibrate,
)
from voice_agent.config import BYTES_PER_FRAME, FRAME_MS, Settings
from voice_agent.events import Emitter, EventSink
from voice_agent.interruptions import (
    InterruptionAssessment,
    InterruptionDecision,
    assess,
)
from voice_agent.prompts_cache import PromptCache
from voice_agent.providers.base import TextDelta, ToolCall, Transcript
from voice_agent.providers.registry import Providers
from voice_agent.session import Session, SpokenTracker, TurnMetrics
from voice_agent.text import ClauseAssembler, clean_for_speech

log = logging.getLogger(__name__)

#: Transcripts below this are treated as "not heard" rather than as words.
#: Whisper reports low average log-probability on breath, line noise, and the
#: hallucinated phrases it produces from near-silence.
MIN_TRANSCRIPT_CONFIDENCE = 0.25

#: Consecutive unusable transcripts before the agent stops apologising and
#: questions the line instead.
MAX_CONSECUTIVE_EMPTY = 2

#: Shorter than this and an unusable transcript is treated as noise rather than
#: as speech that was missed. The clarifier is for "I did not catch that", and
#: a 200ms knock is not something anyone said.
MIN_UTTERANCE_MS = 450

#: Sentinel closing the playback queue.
_END = None


@dataclass(frozen=True, slots=True)
class BargeInCandidate:
    """A complete caller utterance accepted for processing after cancellation."""

    utterance: SpeechEnded
    transcript: Transcript
    assessment: InterruptionAssessment
    asr_ms: float


class Pipeline:
    """Drives one call from calibration to hang-up."""

    def __init__(
        self,
        session: Session,
        providers: Providers,
        turns: TurnSource,
        cache: PromptCache,
        settings: Settings,
        events: EventSink | None = None,
    ) -> None:
        self._session = session
        self._providers = providers
        self._turns = turns
        self._cache = cache
        self._settings = settings
        #: Optional. Defaults to discarding, so the loopback and telephony
        #: transports carry no cost and the turn loop never branches on whether
        #: anyone is watching.
        self._emit = Emitter(events)
        self._detector: UtteranceDetector | None = None
        self._consecutive_empty = 0
        self._preferred_language: locale.Language | None = None

    # ------------------------------------------------------------------ call

    async def run(self) -> None:
        """Run the call until the transport closes."""
        await self._calibrate()

        if self._settings.barge_in:
            log.info("[%s] barge-in armed", self._session.id)

        # Pick from the pre-synthesised time variants at call start, rather
        # than baking the server's startup hour into every later call.
        greeting = self._cache.greeting(locale.current())
        pending = None
        if greeting:
            pending = await self._speak_cached("greeting", greeting)

        while True:
            if pending is None:
                utterance = await self._listen()
                if utterance is None:
                    log.info(
                        "[%s] transport closed after %d turns",
                        self._session.id,
                        self._session.turns,
                    )
                    return
                pending = await self._handle_utterance(utterance)
            else:
                candidate = pending
                pending = await self._handle_utterance(
                    candidate.utterance,
                    transcript=candidate.transcript,
                    asr_ms=candidate.asr_ms,
                )

    async def _calibrate(self) -> None:
        """Measure the ambient noise floor before trusting any threshold.

        A mobile on a street and a desk phone in a quiet office differ by more
        than any fixed threshold can span, so this happens per call.
        """
        frames: list[Frame] = []
        wanted = self._settings.calibration_frames
        while len(frames) < wanted:
            frame = await self._session.channel.get()
            if frame is None:
                break
            frames.append(frame)

        thresholds = calibrate(
            frames,
            min_start=float(self._settings.vad_min_start),
            min_stop=float(self._settings.vad_min_stop),
        )
        if self._settings.start_threshold is not None:
            thresholds = type(thresholds)(
                floor=thresholds.floor,
                start=float(self._settings.start_threshold),
                stop=thresholds.stop,
            )
        if self._settings.stop_threshold is not None:
            thresholds = type(thresholds)(
                floor=thresholds.floor,
                start=thresholds.start,
                stop=float(self._settings.stop_threshold),
            )

        self._session.thresholds = thresholds
        self._detector = UtteranceDetector(
            thresholds,
            onset_frames=self._settings.onset_frames,
            stop_hang_ms=self._settings.stop_hang_ms,
            max_utterance_s=self._settings.max_utterance_s,
        )
        # The barge threshold is logged alongside the rest because these four
        # numbers are what you tune against when the agent is too eager or too
        # deaf, and guessing at them from the outside is hopeless.
        self._emit(
            "calibrated",
            floor=round(thresholds.floor),
            start=round(thresholds.start),
            stop=round(thresholds.stop),
            # Not "Silero". This is an energy VAD whose thresholds are derived
            # from the noise floor measured on this call.
            vad="energy, calibrated",
            endpoint_ms=self._settings.stop_hang_ms,
        )
        log.info(
            "[%s] calibrated floor=%.0f start=%.0f stop=%.0f barge=%.0f",
            self._session.id,
            thresholds.floor,
            thresholds.start,
            thresholds.stop,
            max(
                float(self._settings.barge_in_min_rms),
                thresholds.start * self._settings.echo_threshold_factor,
            ),
        )

    # --------------------------------------------------------------- listen

    async def _listen(self) -> SpeechEnded | None:
        """Collect one utterance, or ``None`` if the call ended."""
        assert self._detector is not None
        self._detector.reset()

        async for frame in self._session.channel.frames():
            now = frame.received_at
            # Inside the guard window this is our own voice coming back.
            if self._session.echo.suppressed(now):
                continue
            event = self._detector.push(frame, scale=self._session.echo.scale(now))
            if isinstance(event, SpeechEnded):
                return event
        return None

    # ----------------------------------------------------------------- turn

    async def _handle_utterance(
        self,
        utterance: SpeechEnded,
        *,
        transcript: Transcript | None = None,
        asr_ms: float | None = None,
    ) -> BargeInCandidate | None:
        metrics = TurnMetrics(endpoint_ms=float(self._settings.stop_hang_ms))
        turn_start = time.monotonic()

        if transcript is None:
            asr_start = time.perf_counter()
            try:
                transcript = await self._providers.asr.transcribe(utterance.pcm)
            except Exception:
                log.exception("[%s] ASR failed", self._session.id)
                metrics.note("asr_error")
                return await self._speak_cached("error", self._cache.get("error", locale.current()))
            metrics.asr_ms = (time.perf_counter() - asr_start) * 1000.0
        else:
            metrics.asr_ms = asr_ms or 0.0

        if transcript.is_empty or transcript.confidence < MIN_TRANSCRIPT_CONFIDENCE:
            # A blip is not somebody talking. A cough, a door, a knock on the
            # microphone all clear the VAD and transcribe to nothing, and
            # apologising for each one makes the agent sound like it is
            # struggling when the line is fine. Below this length there is
            # nothing to have misheard, so say nothing at all.
            if utterance.duration_ms < MIN_UTTERANCE_MS:
                log.debug("[%s] ignoring %dms blip", self._session.id, utterance.duration_ms)
                return None
            return await self._handle_unheard(transcript.confidence, metrics)

        self._consecutive_empty = 0
        self._session.turns += 1
        requested_language = locale.requested(transcript.text)
        if requested_language is not None:
            self._preferred_language = requested_language
        language = requested_language or self._preferred_language or locale.normalise(
            transcript.language
        )
        metrics.language = language
        log.info(
            "[%s] heard %s (%.0f%%, rms=%.0f): %s",
            self._session.id,
            language,
            transcript.confidence * 100,
            wav.rms(utterance.pcm),
            transcript.text,
        )

        self._emit(
            "asr",
            text=transcript.text,
            confidence=round(transcript.confidence, 3),
            language=language,
            ms=round(metrics.asr_ms),
        )

        # Everything downstream — tool output, dates, money, recovery lines, the
        # voice itself — reads the language from here. Set per utterance rather
        # than per call, because a rep switching language mid-conversation is
        # the case this exists for.
        with locale.use(language):
            self._use_voice(language)
            spoken, pending = await self._respond(transcript.text, metrics, turn_start)

        # The single crossing point into conversation state, in one direction.
        # `spoken` is what the caller actually heard, which after an interruption
        # is not what was generated.
        await self._turns.commit(self._session.thread_id, spoken)
        self._session.history.append(metrics)
        # Deterministic, and narrower than it looks: not "was this true" but
        # "did this come from somewhere". The dashboard shows the untraced
        # figures by name, because a count is a score and a name is a prompt to
        # go and look.
        drain = getattr(self._turns, "drain_tool_results", None)
        tool_results = drain() if drain is not None else []
        self._emit(
            "grounding",
            **grounding.summarise(grounding.trace(spoken, tool_results)),
        )
        self._emit(
            "turn_complete",
            turn=self._session.turns,
            spoken=spoken,
            language=metrics.language,
            barged_in=metrics.barged_in,
            generated_chars=metrics.generated_chars,
            spoken_chars=metrics.spoken_chars,
            asr_ms=round(metrics.asr_ms),
            llm_first_token_ms=round(metrics.llm_first_token_ms),
            first_audio_ms=round(metrics.first_audio_ms),
            budget_ms=self._settings.budget.total_ms,
            events=list(metrics.events),
        )
        log.info("[%s] turn %d %s", self._session.id, self._session.turns, metrics.summary())
        return pending

    def _use_voice(self, language: str) -> None:
        """Point the synthesiser at this language's voice, if it can switch.

        Asked through ``getattr`` because it is an optional capability: a cloud
        TTS selects a voice per request and has nothing to switch, and the
        pipeline should not grow a branch per provider.
        """
        switch = getattr(self._providers.tts, "use_language", None)
        if switch is not None:
            switch(language)

    async def _handle_unheard(
        self, confidence: float, metrics: TurnMetrics
    ) -> BargeInCandidate | None:
        """Nothing usable was transcribed.

        The turn is deliberately *not* advanced and nothing is committed to
        history. The implementation this replaces fed the literal string
        ``(inaudible)`` to the model, which poisoned the conversation with a
        user turn the caller never said.
        """
        self._consecutive_empty += 1
        metrics.note(f"unheard({confidence:.2f})")
        name = (
            "clarifier_repeated"
            if self._consecutive_empty >= MAX_CONSECUTIVE_EMPTY
            else "clarifier"
        )
        log.info("[%s] unheard x%d -> %s", self._session.id, self._consecutive_empty, name)
        return await self._speak_cached(name, self._cache.get(name))

    # -------------------------------------------------------------- respond

    async def _respond(
        self, user_text: str, metrics: TurnMetrics, turn_start: float
    ) -> tuple[str, BargeInCandidate | None]:
        """Generate, speak, and return what the caller actually heard."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        tracker = SpokenTracker()
        self._emit("turn_start", turn=self._session.turns, text=user_text)
        first_token = asyncio.Event()
        # Reset here rather than in begin_playback(): a turn cancelled before
        # any audio played would otherwise report the *previous* turn's
        # progress, and commit text the caller never heard.
        self._session.played_bytes = 0

        producer = asyncio.create_task(
            self._produce(user_text, queue, tracker, first_token, metrics, turn_start),
            name="produce",
        )
        player = asyncio.create_task(self._play(queue, metrics, turn_start), name="play")
        speaking: asyncio.Future[Any] = asyncio.gather(producer, player)
        watcher = asyncio.create_task(self._watch_for_barge_in(), name="barge-in")
        racing: set[asyncio.Future[Any]] = {speaking, watcher}
        pending: BargeInCandidate | None = None

        try:
            done, _ = await asyncio.wait(racing, return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                pending = watcher.result()
                if not speaking.done():
                    # Cancel generation, synthesis and playback together. The
                    # browser clears its queue only after this accepted event.
                    metrics.barged_in = True
                    metrics.note("barge_in")
                    self._emit(
                        "barge_in",
                        played_ms=self._session.played_ms,
                        reason=pending.assessment.reason,
                        text=pending.transcript.text,
                    )
                    log.info(
                        "[%s] barge-in at %dms of speech",
                        self._session.id,
                        self._session.played_ms,
                    )
                    speaking.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await speaking
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            self._session.end_playback()
            # Whatever arrived while we were speaking is either our own voice
            # or audio the caller produced over us; neither should be
            # transcribed as their next turn.
            self._session.channel.drain()

        spoken = tracker.spoken(self._session.played_bytes)
        metrics.generated_chars = len(tracker.full_text)
        metrics.spoken_chars = len(spoken)
        return spoken, pending

    async def _produce(
        self,
        user_text: str,
        queue: asyncio.Queue[bytes | None],
        tracker: SpokenTracker,
        first_token: asyncio.Event,
        metrics: TurnMetrics,
        turn_start: float,
    ) -> None:
        """Stream the model, cut it into clauses, synthesise, enqueue."""
        assembler = ClauseAssembler(
            min_chars=self._settings.min_clause_chars,
            flush_after_ms=self._settings.clause_flush_ms,
        )
        guard = asyncio.create_task(
            self._filler_guard(first_token, queue, tracker, metrics), name="filler"
        )
        llm_start = time.perf_counter()
        spoke_anything = False

        try:
            async for delta in self._turns.stream(self._session.thread_id, user_text):
                if isinstance(delta, ToolCall):
                    # The turn source runs tools itself; a call surfacing here
                    # is informational. Never speak it — a function name read
                    # aloud is the worst possible thing to put on a phone line.
                    metrics.note(f"tool:{delta.name}")
                    self._emit("tool_call", name=delta.name, arguments=delta.arguments)
                    continue
                if not isinstance(delta, TextDelta):
                    continue

                if not first_token.is_set():
                    first_token.set()
                    metrics.llm_first_token_ms = (time.perf_counter() - llm_start) * 1000.0

                for clause in assembler.push(delta.text):
                    # Emitted as it is queued, so the page builds the answer up
                    # the way the caller hears it. Sending the whole reply at
                    # the end shows text that has not been spoken yet, which is
                    # the opposite of what a live transcript is for — and after
                    # a barge-in it shows words nobody heard.
                    self._emit("clause", text=clause)
                    if not metrics.first_clause_ms:
                        metrics.first_clause_ms = (
                            time.perf_counter() - llm_start
                        ) * 1000.0 - metrics.llm_first_token_ms
                    spoke_anything |= await self._synthesize(clause, queue, tracker, metrics)

            tail = assembler.flush()
            if tail:
                self._emit("clause", text=tail)
                spoke_anything |= await self._synthesize(tail, queue, tracker, metrics)

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[%s] generation failed", self._session.id)
            metrics.note("llm_error")
            # Salvage buffered text only if it stops somewhere speakable;
            # a fragment cut mid-word sounds like a fault rather than a reply.
            if assembler.ends_at_boundary():
                salvaged = assembler.flush()
                if salvaged:
                    spoke_anything |= await self._synthesize(salvaged, queue, tracker, metrics)
            if not spoke_anything:
                await self._enqueue_cached("error", queue, tracker)
        finally:
            first_token.set()  # release the guard if it is still waiting
            guard.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await guard
            if not spoke_anything and not queue.qsize():
                await self._enqueue_cached("error", queue, tracker)
            await queue.put(_END)

    async def _filler_guard(
        self,
        first_token: asyncio.Event,
        queue: asyncio.Queue[bytes | None],
        tracker: SpokenTracker,
        metrics: TurnMetrics,
    ) -> None:
        """Cover a slow model with a holding line rather than silence."""
        try:
            await asyncio.wait_for(
                first_token.wait(),
                timeout=self._settings.llm_first_token_timeout_ms / 1000.0,
            )
        except TimeoutError:
            metrics.note("filler")
            log.info("[%s] no token in time; playing filler", self._session.id)
            await self._enqueue_cached("filler", queue, tracker)

    async def _synthesize(
        self,
        clause: str,
        queue: asyncio.Queue[bytes | None],
        tracker: SpokenTracker,
        metrics: TurnMetrics,
    ) -> bool:
        """Synthesise one clause into the playback queue.

        A clause that cannot be synthesised is skipped rather than failing the
        turn: a missing phrase is recoverable, a dropped call is not.
        """
        text = clean_for_speech(clause)
        if not text:
            return False

        for attempt in (1, 2):
            started = time.perf_counter()
            audio = bytearray()
            try:
                async for chunk in self._providers.tts.synthesize(text):
                    if not audio and not metrics.tts_first_chunk_ms:
                        metrics.tts_first_chunk_ms = (time.perf_counter() - started) * 1000.0
                    audio += chunk
                    await queue.put(bytes(chunk))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "[%s] TTS attempt %d failed for %r: %s",
                    self._session.id,
                    attempt,
                    text[:40],
                    exc,
                )
                if attempt == 2:
                    metrics.note("tts_skip")
                    return False
                continue

            if audio:
                tracker.add(text, len(audio))
                pad = self._settings.inter_sentence_pad_ms
                if pad:
                    await queue.put(wav.silence(pad))
                return True
            return False
        return False

    async def _enqueue_cached(
        self,
        name: str,
        queue: asyncio.Queue[bytes | None],
        tracker: SpokenTracker,
    ) -> None:
        language = locale.current()
        audio = self._cache.get(name, language)
        if not audio:
            return
        # What the caller heard, in the language they heard it, so an
        # interrupted recovery line commits the words that actually played.
        tracker.add(prompts.line(name, language), len(audio))
        await queue.put(audio)

    # ------------------------------------------------------------- playback

    async def _play(
        self,
        queue: asyncio.Queue[bytes | None],
        metrics: TurnMetrics,
        turn_start: float,
    ) -> None:
        """Send audio at a steady 20ms cadence.

        Paced against a fixed schedule rather than sleeping a flat 20ms per
        frame: the latter accumulates every scheduling delay into permanent
        drift, and a call that drifts is a call whose audio arrives late and
        eventually breaks up.
        """
        buffer = bytearray()
        finished = False
        prebuffer = self._settings.prebuffer_ms * BYTES_PER_FRAME // FRAME_MS

        # Wait for a little audio before starting, so a hiccup in synthesis does
        # not underrun the stream mid-word.
        while len(buffer) < prebuffer and not finished:
            chunk = await queue.get()
            if chunk is _END:
                finished = True
            else:
                buffer += chunk

        if not buffer:
            return

        self._session.begin_playback()
        metrics.first_audio_ms = (time.monotonic() - turn_start) * 1000.0

        loop = asyncio.get_running_loop()
        start = loop.time()
        sent = 0
        try:
            while True:
                while len(buffer) < BYTES_PER_FRAME and not finished:
                    chunk = await queue.get()
                    if chunk is _END:
                        finished = True
                    else:
                        buffer += chunk

                if len(buffer) < BYTES_PER_FRAME:
                    break

                frame = bytes(buffer[:BYTES_PER_FRAME])
                del buffer[:BYTES_PER_FRAME]
                await self._session.transport.send(frame)
                self._session.played_bytes += len(frame)

                sent += 1
                target = start + sent * (FRAME_MS / 1000.0)
                delay = target - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
        finally:
            # Let the tail reach the far end before the mic is trusted again.
            if self._settings.tts_tail_guard_ms:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(self._settings.tts_tail_guard_ms / 1000.0)

    # -------------------------------------------------------------- barge-in

    async def _watch_for_barge_in(self) -> BargeInCandidate:
        """Return only after a speech candidate has been semantically accepted.

        Playback continues while the candidate is transcribed and assessed.
        Energy is evidence that speech should be inspected, not permission to
        cancel the answer.
        """
        if not self._settings.barge_in:
            await asyncio.Event().wait()  # never fires; cancelled with the turn

        threshold = max(
            float(self._settings.barge_in_min_rms),
            self._session.thresholds.start * self._settings.echo_threshold_factor,
        )
        candidate_thresholds = type(self._session.thresholds)(
            floor=self._session.thresholds.floor,
            start=threshold,
            stop=max(self._session.thresholds.stop, threshold * 0.35),
        )
        detector = UtteranceDetector(
            candidate_thresholds,
            onset_frames=self._settings.barge_in_frames,
            stop_hang_ms=self._settings.stop_hang_ms,
            max_utterance_s=self._settings.max_utterance_s,
        )
        # Two independent bars, both of which must be cleared. The relative one
        # adapts to a noisy line; the absolute one is what stops the agent's own
        # voice — leaking past imperfect echo cancellation at a level well above
        # a near-zero measured floor — from reading as an interruption.
        async for frame in self._session.channel.frames():
            if self._session.echo.suppressed(frame.received_at):
                continue
            event = detector.push(frame)
            if not isinstance(event, SpeechEnded):
                continue
            if event.duration_ms < self._settings.barge_in_min_ms:
                self._emit(
                    "barge_candidate",
                    decision=InterruptionDecision.IGNORE,
                    reason="below_minimum_duration",
                    duration_ms=event.duration_ms,
                )
                continue

            asr_start = time.perf_counter()
            try:
                transcript = await self._providers.asr.transcribe(event.pcm)
            except Exception:
                log.exception("[%s] interruption candidate ASR failed", self._session.id)
                self._emit(
                    "barge_candidate",
                    decision=InterruptionDecision.IGNORE,
                    reason="asr_error",
                    duration_ms=event.duration_ms,
                )
                continue
            asr_ms = (time.perf_counter() - asr_start) * 1000.0
            assessment = assess(transcript.text, transcript.confidence)
            self._emit(
                "barge_candidate",
                decision=assessment.decision,
                reason=assessment.reason,
                text=transcript.text,
                confidence=round(transcript.confidence, 3),
                duration_ms=event.duration_ms,
                asr_ms=round(asr_ms),
            )
            if assessment.decision is InterruptionDecision.INTERRUPT:
                log.info(
                    "[%s] semantic barge-in accepted (%s): %s",
                    self._session.id,
                    assessment.reason,
                    transcript.text,
                )
                return BargeInCandidate(event, transcript, assessment, asr_ms)
            log.debug(
                "[%s] semantic barge-in rejected (%s): %s",
                self._session.id,
                assessment.reason,
                transcript.text,
            )

        raise asyncio.CancelledError

    # ------------------------------------------------------------ utilities

    async def _speak_cached(
        self, name: str, audio: bytes | None
    ) -> BargeInCandidate | None:
        """Play one pre-synthesised line, interruptible like any other speech."""
        if not audio:
            log.warning("[%s] cached line %r unavailable", self._session.id, name)
            return None
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        await queue.put(audio)
        await queue.put(_END)
        metrics = TurnMetrics()
        player = asyncio.create_task(self._play(queue, metrics, time.monotonic()))
        watcher = asyncio.create_task(self._watch_for_barge_in())
        racing: set[asyncio.Future[Any]] = {player, watcher}
        pending: BargeInCandidate | None = None
        try:
            done, _ = await asyncio.wait(racing, return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                pending = watcher.result()
                if not player.done():
                    self._emit(
                        "barge_in",
                        played_ms=self._session.played_ms,
                        reason=pending.assessment.reason,
                        text=pending.transcript.text,
                    )
                    player.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await player
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            self._session.end_playback()
            self._session.channel.drain()
        return pending


async def run_call(
    session: Session,
    providers: Providers,
    turns: TurnSource,
    cache: PromptCache,
    settings: Settings,
    events: EventSink | None = None,
) -> None:
    """Run one call to completion, persisting the transcript on the way out."""
    pipeline = Pipeline(session, providers, turns, cache, settings, events)
    try:
        await pipeline.run()
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("[%s] call failed", session.id)
    finally:
        transcript = await turns.history(session.thread_id)
        log.info(
            "[%s] call ended after %.1fs, %d turns, %d messages",
            session.id,
            time.monotonic() - session.started_at,
            session.turns,
            len(transcript),
        )


__all__ = ["Pipeline", "run_call"]
