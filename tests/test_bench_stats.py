"""Tests for the benchmark statistics.

The README publishes whatever these functions compute, so a quiet error here
becomes a false claim in the repository's headline numbers.
"""

from __future__ import annotations

import math

import pytest

from bench.stats import Measurement, render_table, repeat, repeat_reported


class TestMeasurement:
    def test_percentiles_over_known_samples(self) -> None:
        m = Measurement("ASR", "batch", samples=[10.0, 20.0, 30.0, 40.0, 50.0])
        assert m.p50 == 30.0
        assert m.best == 10.0

    def test_p95_tracks_the_tail_not_the_middle(self) -> None:
        # The tail is the point: a good median with an awful p95 is a pipeline
        # that talks over its caller once every twenty turns. One turn in ten
        # stalling must be visible in p95 while leaving p50 untouched.
        m = Measurement("LLM", "ttft", samples=[100.0] * 90 + [2_000.0] * 10)
        assert m.p50 == 100.0
        assert m.p95 == 2_000.0

    def test_empty_samples_are_nan_not_zero(self) -> None:
        # Zero would render as a suspiciously excellent result.
        m = Measurement("TTS", "clause")
        assert math.isnan(m.p50)
        assert math.isnan(m.best)

    def test_unavailable_is_not_ok_even_with_samples(self) -> None:
        m = Measurement("TTS", "clause", samples=[1.0], unavailable="binary missing")
        assert m.ok is False


class TestRepeat:
    def test_records_one_sample_per_run(self) -> None:
        m = repeat(Measurement("x", "y"), lambda: None, runs=4)
        assert len(m.samples) == 4

    def test_warmup_runs_are_not_recorded(self) -> None:
        calls = 0

        def operation() -> None:
            nonlocal calls
            calls += 1

        m = repeat(Measurement("x", "y"), operation, runs=3, warmup=2)
        assert calls == 5
        assert len(m.samples) == 3

    def test_a_failure_marks_unavailable_and_discards_samples(self) -> None:
        # A half-populated result would be averaged into a misleading number.
        calls = 0

        def flaky() -> None:
            nonlocal calls
            calls += 1
            if calls > 3:
                raise RuntimeError("piper died")

        m = repeat(Measurement("TTS", "clause"), flaky, runs=5)
        assert m.ok is False
        assert m.samples == []
        assert "piper died" in (m.unavailable or "")

    def test_failure_during_warmup_is_reported(self) -> None:
        def broken() -> None:
            raise FileNotFoundError("no binary")

        m = repeat(Measurement("TTS", "clause"), broken, runs=3)
        assert m.ok is False
        assert "FileNotFoundError" in (m.unavailable or "")


class TestRepeatReported:
    def test_uses_the_value_the_operation_returns(self) -> None:
        # Wall-clock time around the call includes the settle period, which is
        # setup rather than latency.
        m = repeat_reported(Measurement("TTS", "resident"), lambda: 42.0, runs=3)
        assert m.samples == [42.0, 42.0, 42.0]

    def test_failure_clears_samples(self) -> None:
        calls = 0

        def flaky() -> float:
            nonlocal calls
            calls += 1
            if calls > 2:
                raise RuntimeError("stream ended")
            return 5.0

        m = repeat_reported(Measurement("TTS", "resident"), flaky, runs=5)
        assert m.ok is False
        assert m.samples == []


class TestRenderTable:
    def test_renders_measured_rows(self) -> None:
        m = Measurement("ASR", "batch", samples=[100.0, 200.0])
        table = render_table([m])
        assert "| ASR | batch |" in table
        assert "| 2 |" in table

    def test_unavailable_rows_state_the_reason_rather_than_a_number(self) -> None:
        # An unavailable component must never render as a plausible timing.
        m = Measurement("ASR", "batch", unavailable="faster-whisper not installed")
        table = render_table([m])
        assert "faster-whisper not installed" in table
        assert "nan" not in table.lower()

    def test_handles_an_empty_run(self) -> None:
        assert "Component" in render_table([])


class TestBudgetGuard:
    def test_percentile_of_single_sample(self) -> None:
        m = Measurement("x", "y", samples=[7.0])
        assert m.p50 == pytest.approx(7.0)
        assert m.p95 == pytest.approx(7.0)
