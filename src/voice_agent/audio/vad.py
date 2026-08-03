"""Energy VAD, echo gating and barge-in detection.

These three live in one module because they are one subsystem answering one
question: *is the energy arriving right now the caller, or our own voice coming
back down the line?* The codebase this replaces split them across a socket loop,
a playback loop and a pile of module-level globals, which is why its barge-in
shipped disabled.

Everything here is a pure state machine — frames in, events out. No sockets, no
tasks, no wall clock. That is what makes interruption testable without standing
up a call.

Two deliberate departures from the original implementation:

1. **Pre-roll.** The original began buffering at *confirmed* onset, discarding
   the consecutive loud frames that proved speech had started — 120 ms of the
   caller's first syllable, thrown away on every turn. Short answers ("yes",
   "no", a postcode) lost their attack and transcribed poorly. A ring buffer now
   retains the frames leading up to onset and prepends them.

2. **Frame counting instead of wall clock.** The original measured the
   end-of-utterance hangover with ``time.time()``, so network jitter or a slow
   event loop iteration could stall the timer and hold the turn open, or a burst
   of buffered frames could collapse it and cut the caller off. Frames are
   exactly 20 ms of audio by construction, so counting them measures *audio*
   time rather than *processing* time. It is both more correct under load and
   deterministic to test.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from voice_agent.audio.framing import Frame
from voice_agent.config import FRAME_MS

#: Multipliers applied to the measured noise floor. Speech must clear the floor
#: by a healthy margin to start a turn, but is allowed to decay closer to it
#: before we call the turn over — hysteresis, so a quiet syllable mid-sentence
#: does not end the utterance.
_START_FACTOR: Final = 2.0
_STOP_FACTOR: Final = 1.35

#: Fallback floors, used when a caller does not supply their own.
#:
#: These matter far more than they look. Scaling to the measured noise floor
#: assumes the floor is a real signal, which holds on a phone line and fails
#: completely on a browser with noise suppression, where the floor is close to
#: zero and the scaled threshold collapses onto these values. Set them near
#: silence — as the implementation this replaces did, at 12 and 8 — and every
#: frame reads as speech: the microphone triggers on nothing, and the agent
#: interrupts itself on its own voice returning from the speakers.
#:
#: On the 0..32767 RMS scale, quiet room tone sits below ~50 and ordinary
#: speech lands in the high hundreds to low thousands.
_MIN_START: Final = 280.0
_MIN_STOP: Final = 160.0

#: Frames of lead-in retained ahead of confirmed onset, on top of the onset
#: window itself. 100 ms, enough to catch the soft attack of a leading fricative.
_PREROLL_EXTRA_FRAMES: Final = 5


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Energy thresholds derived from the ambient noise floor of this call."""

    floor: float
    start: float
    stop: float

    def scaled(self, factor: float) -> Thresholds:
        """Both thresholds raised, used while our own voice may still be echoing."""
        return Thresholds(floor=self.floor, start=self.start * factor, stop=self.stop * factor)


def calibrate(
    frames: list[Frame],
    *,
    min_start: float = _MIN_START,
    min_stop: float = _MIN_STOP,
) -> Thresholds:
    """Derive thresholds from ambient audio sampled at the start of a call.

    Calibrating per call matters: a mobile on a street and a desk phone in a
    quiet office differ by more than any fixed threshold can span. But the
    floors matter just as much — on a clean input the measured floor carries no
    information, and the minimums are what actually separate speech from
    silence.
    """
    if not frames:
        return Thresholds(floor=0.0, start=min_start, stop=min_stop)
    floor = sum(f.rms for f in frames) / len(frames)
    return Thresholds(
        floor=floor,
        start=max(min_start, floor * _START_FACTOR),
        stop=max(min_stop, floor * _STOP_FACTOR),
    )


class ConsecutiveTrigger:
    """Fires once N consecutive frames clear a threshold; any quiet frame resets it.

    The shared primitive behind both "the caller started talking" and "the caller
    is interrupting". Requiring a *run* of loud frames rather than a single one is
    what separates speech from a door slam, a keyboard, or a codec artefact.
    """

    __slots__ = ("_count", "_needed")

    def __init__(self, needed: int) -> None:
        if needed < 1:
            raise ValueError(f"needed must be >= 1, got {needed}")
        self._needed = needed
        self._count = 0

    def push(self, value: float, threshold: float) -> bool:
        """Return ``True`` on the frame that completes the run."""
        if value > threshold:
            self._count += 1
            return self._count >= self._needed
        self._count = 0
        return False

    def reset(self) -> None:
        self._count = 0

    @property
    def progress(self) -> int:
        return self._count


class EndReason(StrEnum):
    SILENCE = "silence"
    MAX_DURATION = "max_duration"


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """The caller began speaking; ``at_seq`` is the frame that confirmed it."""

    at_seq: int


@dataclass(frozen=True, slots=True)
class SpeechEnded:
    """A complete utterance, ready for ASR."""

    pcm: bytes
    reason: EndReason
    duration_ms: int


VadEvent = SpeechStarted | SpeechEnded


class UtteranceDetector:
    """Segments a frame stream into utterances.

    Feed every frame; act on the events returned. The detector owns no audio
    beyond the utterance currently in flight and the pre-roll ring.
    """

    def __init__(
        self,
        thresholds: Thresholds,
        *,
        onset_frames: int,
        stop_hang_ms: int,
        max_utterance_s: float,
    ) -> None:
        self._thresholds = thresholds
        self._onset = ConsecutiveTrigger(onset_frames)
        self._stop_hang_frames = max(1, stop_hang_ms // FRAME_MS)
        self._max_frames = max(1, int(max_utterance_s * 1000) // FRAME_MS)
        self._preroll: deque[Frame] = deque(maxlen=onset_frames + _PREROLL_EXTRA_FRAMES)

        self._speaking = False
        self._buffer = bytearray()
        self._quiet_frames = 0
        self._voiced_frames = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def thresholds(self) -> Thresholds:
        return self._thresholds

    def set_thresholds(self, thresholds: Thresholds) -> None:
        """Adjust thresholds mid-call, e.g. while echo may still be present."""
        self._thresholds = thresholds

    def push(self, frame: Frame, *, scale: float = 1.0) -> VadEvent | None:
        """Advance the state machine by one frame.

        ``scale`` raises both thresholds for this frame only; the echo gate uses
        it to be temporarily suspicious of energy right after the agent speaks,
        without permanently desensitising the detector.
        """
        start_threshold = self._thresholds.start * scale
        stop_threshold = self._thresholds.stop * scale

        if not self._speaking:
            self._preroll.append(frame)
            if self._onset.push(frame.rms, start_threshold):
                return self._begin(frame)
            return None

        self._buffer.extend(frame.pcm)
        self._voiced_frames += 1

        if frame.rms > stop_threshold:
            self._quiet_frames = 0
        else:
            self._quiet_frames += 1
            if self._quiet_frames >= self._stop_hang_frames:
                return self._end(EndReason.SILENCE)

        if self._voiced_frames >= self._max_frames:
            return self._end(EndReason.MAX_DURATION)
        return None

    def _begin(self, frame: Frame) -> SpeechStarted:
        self._speaking = True
        self._quiet_frames = 0
        # Prepend the lead-in, including the frames that proved onset. Without
        # this the utterance starts mid-syllable.
        self._buffer = bytearray(b"".join(f.pcm for f in self._preroll))
        self._voiced_frames = len(self._preroll)
        self._preroll.clear()
        return SpeechStarted(at_seq=frame.seq)

    def _end(self, reason: EndReason) -> SpeechEnded:
        pcm = bytes(self._buffer)
        event = SpeechEnded(pcm=pcm, reason=reason, duration_ms=self._voiced_frames * FRAME_MS)
        self.reset()
        return event

    def reset(self) -> None:
        """Abandon any utterance in flight and return to listening."""
        self._speaking = False
        self._buffer = bytearray()
        self._quiet_frames = 0
        self._voiced_frames = 0
        self._onset.reset()
        self._preroll.clear()


@dataclass
class EchoGate:
    """Decides how much to trust inbound energy, given what we were just doing.

    Telephony gives us a mixed signal with no acoustic echo cancellation, so our
    own speech arrives back on the inbound stream. Three regimes, in decreasing
    severity:

    * **suppressed** — within the guard window after playback. Frames are dropped
      outright; nothing they contain can be trusted.
    * **raised** — for a further window, thresholds are multiplied so residual
      tail energy cannot masquerade as speech.
    * **open** — normal sensitivity.

    The browser transport negotiates real AEC via ``getUserMedia``, so the guard
    is configured near zero there; the telephony path leans on it heavily.
    """

    guard_ms: int
    raised_factor: float
    #: How long the raised-threshold regime lasts beyond the hard guard window.
    raised_ms: int = 1_000
    _playback_ended_at: float | None = field(default=None, repr=False)

    def on_playback_end(self, at: float) -> None:
        self._playback_ended_at = at

    def on_playback_start(self) -> None:
        # While speaking there is no "since playback ended"; the pipeline gates
        # on its own speaking flag, and this keeps the window from firing early.
        self._playback_ended_at = None

    def suppressed(self, now: float) -> bool:
        """True while inbound audio must be dropped entirely."""
        if self._playback_ended_at is None:
            return False
        return (now - self._playback_ended_at) * 1000.0 < self.guard_ms

    def scale(self, now: float) -> float:
        """Threshold multiplier appropriate to this moment."""
        if self._playback_ended_at is None:
            return 1.0
        elapsed_ms = (now - self._playback_ended_at) * 1000.0
        if elapsed_ms < self.guard_ms + self.raised_ms:
            return self.raised_factor
        return 1.0
