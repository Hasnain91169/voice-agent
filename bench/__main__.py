"""Run the component latency benchmarks and report against the budget.

    uv run python -m bench --runs 5

Phase 1 exists to answer one question before any pipeline is written: on this
hardware, with this stack, is the published 800 ms first-audio target reachable?
The answer sets the provider defaults for both profiles.
"""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass

from bench.probes import probe_asr, probe_llm, probe_tts, probe_tts_resident
from bench.stats import Measurement, render_table
from voice_agent.config import LatencyBudget, Settings


@dataclass
class Estimate:
    """A first-audio estimate assembled from measured stages."""

    stages: dict[str, float]
    modelled: set[str]

    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())

    def render(self, budget: LatencyBudget) -> str:
        lines = [
            "| Stage | Budget (ms) | Measured (ms) | |",
            "|---|---:|---:|:--|",
        ]
        budget_by_stage = {
            "endpoint detection": budget.endpoint_detection,
            "ASR": budget.asr_finalise,
            "LLM first token": budget.llm_first_token,
            "first clause": budget.first_clause,
            "TTS first chunk": budget.tts_first_chunk,
            "prebuffer": budget.prebuffer,
        }
        for stage, actual in self.stages.items():
            allowed = budget_by_stage.get(stage, 0)
            note = "modelled" if stage in self.modelled else ""
            flag = "" if actual <= allowed else " ⚠"
            lines.append(f"| {stage} | {allowed} | {actual:.0f}{flag} | {note} |")
        verdict = "within budget" if self.total_ms <= budget.total_ms else "OVER BUDGET"
        lines.append(
            f"| **total** | **{budget.total_ms}** | **{self.total_ms:.0f}** | **{verdict}** |"
        )
        return "\n".join(lines)


def estimate_first_audio(measurements: list[Measurement], settings: Settings) -> Estimate | None:
    """Assemble a first-audio estimate from whatever was measured.

    Returns ``None`` if a stage on the critical path could not be measured — a
    partial estimate would be worse than none, because it would understate the
    total and look like good news.
    """
    asr = next((m for m in measurements if m.component.startswith("ASR") and m.ok), None)
    llm = next((m for m in measurements if m.component.startswith("LLM") and m.ok), None)
    # Prefer the resident measurement: it reflects the adapter we will actually
    # ship, whereas the spawn-per-clause figure describes the design being replaced.
    tts = next(
        (
            m
            for m in measurements
            if "resident" in m.component and "short clause" in m.label and m.ok
        ),
        None,
    ) or next(
        (
            m
            for m in measurements
            if m.component.startswith("TTS") and m.label == "short clause" and m.ok
        ),
        None,
    )
    if asr is None or llm is None or tts is None:
        return None

    return Estimate(
        stages={
            "endpoint detection": float(settings.stop_hang_ms),
            "ASR": asr.p50,
            "LLM first token": llm.p50,
            # Not separately measurable without the pipeline: the time from the
            # first token to a clause worth speaking. Taken from the budget and
            # marked as modelled so the total is not mistaken for fully measured.
            "first clause": float(settings.budget.first_clause),
            "TTS first chunk": tts.p50,
            "prebuffer": float(settings.prebuffer_ms),
        },
        modelled={"first clause"},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="timed runs per probe")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--asr-device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="faster-whisper device; measure both to justify the deployment target",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=["tts", "llm", "asr"],
        help="skip a probe (repeatable)",
    )
    args = parser.parse_args(argv)

    # The Windows console defaults to cp1252, which cannot encode the em-dashes
    # and warning glyphs in the report.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = Settings()
    print(f"# Component latency — {platform.node()}")
    print()
    print(f"- platform: {platform.platform()}")
    print(f"- python: {sys.version.split()[0]}")
    print(f"- profile: {settings.profile}")
    print(f"- runs per probe: {args.runs} (plus 1 discarded warm-up)")
    print()

    measurements: list[Measurement] = []
    if "tts" not in args.skip:
        measurements += probe_tts(args.runs)
        measurements += probe_tts_resident(args.runs)
    if "llm" not in args.skip:
        measurements += probe_llm(args.runs, args.ollama_url)
    if "asr" not in args.skip:
        measurements += probe_asr(args.runs, args.asr_device)

    print(render_table(measurements))
    print()

    estimate = estimate_first_audio(measurements, settings)
    print("## First-audio estimate")
    print()
    if estimate is None:
        print(
            "Not enough of the critical path was measurable to estimate "
            "first-audio latency. Stages missing above must be resolved first — "
            "a partial total would understate the real figure."
        )
        return 1

    print(estimate.render(settings.budget))
    print()
    if estimate.total_ms > settings.budget.total_ms:
        over = estimate.total_ms - settings.budget.total_ms
        print(
            f"This stack misses the {settings.budget.total_ms} ms target by "
            f"{over:.0f} ms. That is a finding, not a failure: it is why two "
            f"profiles exist and why the local one publishes its real number."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
