"""Timing primitives for the component benchmarks.

Percentiles rather than means: a voice pipeline is judged on its bad turns. A
mean hides the 1-in-20 turn where the model stalls and the caller starts talking
over the agent, which is exactly the turn that ruins a call.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Measurement:
    """Repeated timings of one operation, in milliseconds."""

    component: str
    label: str
    samples: list[float] = field(default_factory=list)
    #: Set when the component could not be measured, explaining why.
    unavailable: str | None = None

    def add(self, elapsed_ms: float) -> None:
        self.samples.append(elapsed_ms)

    @property
    def ok(self) -> bool:
        return self.unavailable is None and bool(self.samples)

    def percentile(self, p: float) -> float:
        if not self.samples:
            return float("nan")
        return float(np.percentile(self.samples, p))

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def best(self) -> float:
        return min(self.samples) if self.samples else float("nan")


@contextmanager
def timed(measurement: Measurement) -> Iterator[None]:
    """Record one timing into ``measurement``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        measurement.add((time.perf_counter() - start) * 1000.0)


def repeat(
    measurement: Measurement,
    operation: Callable[[], object],
    *,
    runs: int,
    warmup: int = 1,
) -> Measurement:
    """Run an operation, discarding warm-up iterations.

    Warm-up matters disproportionately here: the first Piper invocation pays
    ONNX graph planning, the first Ollama call pays model load, and the first
    Whisper call pays CUDA context creation. A caller never experiences those on
    a warm server, so including them would slander the steady state — which is
    also why the pipeline warms every provider at startup.
    """
    for _ in range(warmup):
        try:
            operation()
        except Exception as exc:
            measurement.unavailable = f"{type(exc).__name__}: {exc}"
            return measurement

    for _ in range(runs):
        try:
            with timed(measurement):
                operation()
        except Exception as exc:
            measurement.unavailable = f"{type(exc).__name__}: {exc}"
            measurement.samples.clear()
            return measurement
    return measurement


def repeat_reported(
    measurement: Measurement,
    operation: Callable[[], float],
    *,
    runs: int,
    warmup: int = 1,
) -> Measurement:
    """Like :func:`repeat`, but the operation reports its own elapsed time.

    Needed where wall-clock time around the call is not the quantity of interest
    — the resident-TTS probe must settle the previous utterance before issuing
    the next request, and that settling time is setup, not latency.
    """
    for _ in range(warmup):
        try:
            operation()
        except Exception as exc:
            measurement.unavailable = f"{type(exc).__name__}: {exc}"
            return measurement

    for _ in range(runs):
        try:
            measurement.add(operation())
        except Exception as exc:
            measurement.unavailable = f"{type(exc).__name__}: {exc}"
            measurement.samples.clear()
            return measurement
    return measurement


def render_table(measurements: list[Measurement]) -> str:
    """Format results as a markdown table for the README."""
    header = (
        "| Component | Operation | p50 (ms) | p95 (ms) | best (ms) | n |\n"
        "|---|---|---:|---:|---:|---:|"
    )
    rows = []
    for m in measurements:
        if not m.ok:
            reason = m.unavailable or "no samples"
            rows.append(f"| {m.component} | {m.label} | — | — | — | _{reason}_ |")
        else:
            rows.append(
                f"| {m.component} | {m.label} | {m.p50:.0f} | {m.p95:.0f} "
                f"| {m.best:.0f} | {len(m.samples)} |"
            )
    return "\n".join([header, *rows])
