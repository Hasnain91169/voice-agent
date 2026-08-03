"""Tests for the VAD, echo gate and the shared consecutive-frame trigger.

These are the behaviours that decide whether a call feels natural or maddening,
so they are tested at the state-machine level where every case is reachable.
"""

from __future__ import annotations

import pytest

from tests.conftest import make_frame, square_pcm
from voice_agent.audio.vad import (
    ConsecutiveTrigger,
    EchoGate,
    EndReason,
    SpeechEnded,
    SpeechStarted,
    Thresholds,
    UtteranceDetector,
    calibrate,
)
from voice_agent.config import BYTES_PER_FRAME, FRAME_MS

QUIET = 10
LOUD = 500
THRESHOLDS = Thresholds(floor=20.0, start=100.0, stop=50.0)


def build_detector(
    *,
    onset_frames: int = 6,
    stop_hang_ms: int = 240,
    max_utterance_s: float = 1.0,
) -> UtteranceDetector:
    return UtteranceDetector(
        THRESHOLDS,
        onset_frames=onset_frames,
        stop_hang_ms=stop_hang_ms,
        max_utterance_s=max_utterance_s,
    )


def feed(detector: UtteranceDetector, amplitudes: list[int], *, start_seq: int = 0) -> list[object]:
    events = []
    for i, amplitude in enumerate(amplitudes):
        event = detector.push(make_frame(amplitude, start_seq + i))
        if event is not None:
            events.append(event)
    return events


class TestCalibrate:
    def test_a_clean_input_still_gets_speech_level_thresholds(self) -> None:
        """Regression: the bug that made the agent interrupt itself.

        A browser with noise suppression sends a floor near zero. Scaling to it
        collapses the thresholds onto the minimums, so if those sit near silence
        every frame reads as speech — the microphone triggers on nothing and the
        agent's own voice off the speakers reads as an interruption. The
        minimums have to be at speech level, not at silence level.
        """
        thresholds = calibrate([make_frame(0, i) for i in range(25)])
        assert thresholds.start >= 200.0
        assert thresholds.stop >= 100.0

    def test_floors_are_configurable(self) -> None:
        thresholds = calibrate([make_frame(1, i) for i in range(10)], min_start=500, min_stop=300)
        assert thresholds.start == 500.0
        assert thresholds.stop == 300.0

    def test_a_loud_room_still_scales_above_the_floor(self) -> None:
        # The minimums must not defeat calibration where the floor is real.
        thresholds = calibrate([make_frame(900, i) for i in range(25)])
        assert thresholds.start > 900.0

    def test_no_frames_still_returns_usable_thresholds(self) -> None:
        thresholds = calibrate([])
        assert thresholds.start > 0 and thresholds.stop > 0

    def test_scales_with_the_measured_noise_floor(self) -> None:
        # A caller on a noisy street needs higher thresholds than one in an
        # office; that is the whole point of calibrating per call.
        quiet = calibrate([make_frame(20, i) for i in range(25)])
        noisy = calibrate([make_frame(400, i) for i in range(25)])
        assert noisy.start > quiet.start
        assert noisy.stop > quiet.stop

    def test_start_is_above_stop_for_hysteresis(self) -> None:
        # Speech must clear a higher bar to begin than to continue, so a quiet
        # syllable mid-sentence does not end the turn.
        thresholds = calibrate([make_frame(300, i) for i in range(25)])
        assert thresholds.start > thresholds.stop

    def test_scaled_raises_both_thresholds(self) -> None:
        scaled = THRESHOLDS.scaled(2.0)
        assert scaled.start == 200.0
        assert scaled.stop == 100.0
        assert scaled.floor == THRESHOLDS.floor


class TestConsecutiveTrigger:
    def test_fires_on_the_nth_consecutive_frame(self) -> None:
        trigger = ConsecutiveTrigger(3)
        assert trigger.push(500, 100) is False
        assert trigger.push(500, 100) is False
        assert trigger.push(500, 100) is True

    def test_a_single_quiet_frame_resets_the_run(self) -> None:
        # This is what rejects a door slam or a codec artefact.
        trigger = ConsecutiveTrigger(3)
        trigger.push(500, 100)
        trigger.push(500, 100)
        trigger.push(10, 100)
        assert trigger.progress == 0
        assert trigger.push(500, 100) is False

    def test_rejects_a_nonsense_threshold_count(self) -> None:
        with pytest.raises(ValueError):
            ConsecutiveTrigger(0)


class TestOnset:
    def test_requires_a_full_run_of_loud_frames(self) -> None:
        detector = build_detector(onset_frames=6)
        assert feed(detector, [LOUD] * 5) == []
        assert detector.speaking is False

    def test_starts_on_the_frame_that_completes_the_run(self) -> None:
        detector = build_detector(onset_frames=6)
        events = feed(detector, [LOUD] * 6)
        assert len(events) == 1
        assert isinstance(events[0], SpeechStarted)
        assert events[0].at_seq == 5
        assert detector.speaking is True

    def test_an_isolated_burst_does_not_start_a_turn(self) -> None:
        detector = build_detector(onset_frames=6)
        assert feed(detector, [LOUD, LOUD, QUIET, LOUD, LOUD, LOUD]) == []

    def test_raised_scale_suppresses_onset(self) -> None:
        # The echo gate uses this to distrust energy right after we spoke.
        detector = build_detector(onset_frames=3)
        for _ in range(10):
            assert detector.push(make_frame(LOUD), scale=6.0) is None
        assert detector.speaking is False


class TestPreRoll:
    """Regression tests for the clipped-first-syllable bug.

    The original implementation began buffering at *confirmed* onset, throwing
    away the consecutive loud frames that proved speech had started. Short
    answers lost their attack and transcribed badly.
    """

    def test_utterance_contains_audio_from_before_confirmed_onset(self) -> None:
        detector = build_detector(onset_frames=6)
        first_loud = square_pcm(LOUD)

        feed(detector, [QUIET] * 3 + [LOUD] * 6)
        events = feed(detector, [QUIET] * 12, start_seq=9)

        ended = events[-1]
        assert isinstance(ended, SpeechEnded)
        # Under the old behaviour only the sixth loud frame survived, so the
        # first one would be absent from the captured utterance.
        assert first_loud in ended.pcm

    def test_all_onset_frames_are_retained(self) -> None:
        detector = build_detector(onset_frames=6)
        feed(detector, [QUIET] * 3 + [LOUD] * 6)
        events = feed(detector, [QUIET] * 12, start_seq=9)

        ended = events[-1]
        assert isinstance(ended, SpeechEnded)
        # 3 lead-in + 6 onset frames captured before speech was even confirmed,
        # plus the 12 trailing quiet frames that ended it.
        assert len(ended.pcm) == 21 * BYTES_PER_FRAME

    def test_preroll_is_bounded(self) -> None:
        # A long silence before speech must not accumulate unboundedly.
        detector = build_detector(onset_frames=6)
        feed(detector, [QUIET] * 500)
        feed(detector, [LOUD] * 6, start_seq=500)
        events = feed(detector, [QUIET] * 12, start_seq=506)

        ended = events[-1]
        assert isinstance(ended, SpeechEnded)
        # Ring capacity is onset_frames + 5, not the 500 frames of preceding hush.
        assert len(ended.pcm) == (11 + 12) * BYTES_PER_FRAME


class TestEndOfUtterance:
    def test_ends_after_the_configured_hangover(self) -> None:
        detector = build_detector(onset_frames=6, stop_hang_ms=240)
        feed(detector, [LOUD] * 6)

        assert feed(detector, [QUIET] * 11, start_seq=6) == []
        events = feed(detector, [QUIET], start_seq=17)
        assert isinstance(events[0], SpeechEnded)
        assert events[0].reason is EndReason.SILENCE

    def test_a_loud_frame_resets_the_hangover(self) -> None:
        # Someone pausing mid-sentence must not have their turn ended.
        detector = build_detector(onset_frames=6, stop_hang_ms=240)
        feed(detector, [LOUD] * 6)
        feed(detector, [QUIET] * 11, start_seq=6)
        feed(detector, [LOUD], start_seq=17)

        assert feed(detector, [QUIET] * 11, start_seq=18) == []
        assert detector.speaking is True

    def test_hangover_is_measured_in_audio_not_wall_clock(self) -> None:
        # Frames are 20ms of audio by construction, so a stalled event loop or a
        # burst of buffered frames cannot distort the endpoint decision. All the
        # frames here claim to have arrived at the same instant.
        detector = build_detector(onset_frames=6, stop_hang_ms=240)
        for i in range(6):
            detector.push(make_frame(LOUD, i, received_at=0.0))

        events = [detector.push(make_frame(QUIET, 6 + i, received_at=0.0)) for i in range(12)]
        assert isinstance(events[-1], SpeechEnded)
        assert events[-1].duration_ms == 18 * FRAME_MS

    def test_max_duration_caps_a_monologue(self) -> None:
        detector = build_detector(onset_frames=6, max_utterance_s=1.0)
        events = feed(detector, [LOUD] * 60)
        ended = [e for e in events if isinstance(e, SpeechEnded)]
        assert ended and ended[0].reason is EndReason.MAX_DURATION

    def test_detector_returns_to_listening_after_an_utterance(self) -> None:
        detector = build_detector(onset_frames=6, stop_hang_ms=240)
        feed(detector, [LOUD] * 6)
        feed(detector, [QUIET] * 12, start_seq=6)
        assert detector.speaking is False

        events = feed(detector, [LOUD] * 6, start_seq=18)
        assert isinstance(events[0], SpeechStarted)

    def test_reset_abandons_the_utterance_in_flight(self) -> None:
        detector = build_detector(onset_frames=6)
        feed(detector, [LOUD] * 6)
        detector.reset()
        assert detector.speaking is False
        assert feed(detector, [QUIET] * 20, start_seq=6) == []


class TestEchoGate:
    def test_open_before_any_playback(self) -> None:
        gate = EchoGate(guard_ms=250, raised_factor=1.6)
        assert gate.suppressed(now=100.0) is False
        assert gate.scale(now=100.0) == 1.0

    def test_suppresses_inside_the_guard_window(self) -> None:
        # Our own voice is still arriving; nothing in it can be trusted.
        gate = EchoGate(guard_ms=250, raised_factor=1.6)
        gate.on_playback_end(at=10.0)
        assert gate.suppressed(now=10.1) is True

    def test_raises_thresholds_after_the_guard_window(self) -> None:
        gate = EchoGate(guard_ms=250, raised_factor=1.6, raised_ms=1_000)
        gate.on_playback_end(at=10.0)
        assert gate.suppressed(now=10.4) is False
        assert gate.scale(now=10.4) == 1.6

    def test_returns_to_normal_sensitivity_eventually(self) -> None:
        gate = EchoGate(guard_ms=250, raised_factor=1.6, raised_ms=1_000)
        gate.on_playback_end(at=10.0)
        assert gate.scale(now=12.0) == 1.0

    def test_playback_start_clears_the_window(self) -> None:
        gate = EchoGate(guard_ms=250, raised_factor=1.6)
        gate.on_playback_end(at=10.0)
        gate.on_playback_start()
        assert gate.suppressed(now=10.1) is False

    def test_zero_guard_suppresses_nothing(self) -> None:
        # The browser transport negotiates real AEC, so the guard is unnecessary
        # there and costs latency if left on.
        gate = EchoGate(guard_ms=0, raised_factor=1.0)
        gate.on_playback_end(at=10.0)
        assert gate.suppressed(now=10.0) is False
