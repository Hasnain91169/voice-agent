"""Run the evaluation suite.

    uv run python -m evals                    # text mode, every scenario
    uv run python -m evals --audio            # through the real pipeline
    uv run python -m evals --only 02          # one scenario
    uv run python -m evals --repeat 5         # report a pass rate, not a sample
    uv run python -m evals --transcripts      # print the dialogue for failures

Exits non-zero if any scenario's pass rate falls below ``--min-pass-rate``,
so it can gate a merge.

**On repeating.** A suite scored partly by an LLM judge is not deterministic,
and a single run of it is a sample rather than a measurement. Five consecutive
runs of this suite produced 7/7 four times and 6/7 once — with a *different*
scenario failing each time it failed. Reading any one of those runs as "the
score" would have been wrong in both directions: it would have credited a fix
that changed nothing, and condemned a scenario that was fine. ``--repeat``
exists so the number reported is a rate over N runs, and so that chasing a
green tick by re-rolling is visibly not an option.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import platform
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evals.harness import run_audio, run_text
from evals.judge import Verdict
from evals.scenario import Scenario, load_all
from voice_agent.agent.tools.db import seed
from voice_agent.config import LatencyBudget, Settings
from voice_agent.providers.base import LLM
from voice_agent.providers.registry import build_providers


def render(verdicts: list[Verdict], *, audio: bool) -> str:
    header = "| Scenario | Turns | Tools | Facts | Handled | Goal | Grounded | Result |"
    rule = "|---|--:|:--:|:--:|:--:|:--:|:--:|:--:|"
    if audio:
        header = (
            "| Scenario | Turns | First audio p50 | Tools | Facts | Handled "
            "| Goal | Grounded | Result |"
        )
        rule = "|---|--:|--:|:--:|:--:|:--:|:--:|:--:|:--:|"

    def tick(value: bool | None, *, inverted: bool = False) -> str:
        if value is None:
            return "?"
        wanted = not value if inverted else value
        mark = "yes" if value else "no"
        # A refusal scenario shows 'no (wanted)' rather than a bare failure.
        suffix = "" if wanted else "**"
        if inverted:
            return f"{mark} (refused)" if not value else "**gave in**"
        return f"{suffix}{mark}{suffix}"

    lines = [header, rule]
    for v in verdicts:
        tools = (
            f"{len(v.tools_expected) - len(v.missing_tools)}/{len(v.tools_expected)}"
            if v.tools_expected
            else "-"
        )
        facts = f"{v.facts_found}/{v.facts_expected}" if v.facts_expected else "-"
        result = "pass" if v.passed else "**FAIL**"
        if v.skipped:
            result = f"skip ({v.skipped})"
        if v.error:
            result = f"**ERROR** {v.error[:40]}"
        cells = [v.scenario, str(v.turns)]
        if audio:
            median = f"{statistics.median(v.first_audio_ms):.0f}ms" if v.first_audio_ms else "-"
            cells.append(median)
        cells += [
            tools,
            facts,
            tick(v.handled_well),
            # Reported, not gating: an honest 'no record of that' leaves the
            # caller unsatisfied and the agent blameless.
            tick(v.goal_met, inverted=v.goal_should_fail),
            tick(v.grounded),
            result,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarise_barge_in(verdicts: list[Verdict]) -> str:
    """Three separate facts, because two of them were being conflated.

    An interruption *injected* is not a barge-in. The timing recorded here is
    the gap between the caller starting to talk over the agent and the agent's
    last outbound frame — and an agent that simply finished its sentence
    produces exactly the same number as one that was cancelled mid-clause.
    Reporting that as "barge-in p50" claimed the mechanism had fired on evidence
    that could not tell it apart from the mechanism never running at all.

    So: how many interruptions went in, how many turns the pipeline actually
    cancelled, and of those how many kept history honest.
    """
    samples = sorted(ms for v in verdicts for ms in v.barge_in_ms)
    if not samples:
        return ""
    cancelled = sum(v.barge_ins for v in verdicts)
    checked = [v for v in verdicts if v.commit_truncated is not None]
    honest = sum(1 for v in checked if v.commit_truncated)

    line = f"\n{len(samples)} interruptions injected; the pipeline cancelled {cancelled} turn(s)."
    if cancelled:
        line += (
            f" Caller talking over the agent to its last frame: p50 "
            f"{statistics.median(samples):.0f}ms, worst {samples[-1]:.0f}ms."
        )
    else:
        line += (
            " **No turn was cancelled**, so these timings measure the agent"
            " finishing normally, not barge-in."
        )
    if checked:
        line += (
            f" History matched what the caller actually heard in {honest} of "
            f"{len(checked)} interrupted calls."
        )
    return line + "\n"


def summarise_latency(verdicts: list[Verdict], budget: LatencyBudget) -> str:
    """First-audio percentiles, or a refusal to report them.

    The guard is not decoration. An earlier version measured to the agent's
    *last* frame rather than its first, went negative whenever the agent spoke
    over the caller, and printed "p50 -3781ms — within budget". A budget check
    that accepts a negative latency is not checking anything.
    """
    samples = sorted(ms for v in verdicts for ms in v.first_audio_ms)
    overlaps = sum(v.overlaps for v in verdicts)
    if not samples:
        return "\nNo first-audio samples — every turn overlapped.\n" if overlaps else ""
    if samples[0] < 0:  # pragma: no cover - impossible by construction now
        return "\n**First-audio measurement is broken**: negative samples present.\n"

    p50 = statistics.median(samples)
    p95 = samples[max(0, int(len(samples) * 0.95) - 1)]
    verdict = "within budget" if p50 <= budget.total_ms else "**over budget**"
    line = (
        f"\nFirst audio across {len(samples)} turns, measured from outside the "
        f"process: p50 {p50:.0f}ms, p95 {p95:.0f}ms, against a {budget.total_ms}ms "
        f"target — {verdict}."
    )
    if overlaps:
        line += (
            f" {overlaps} further turns excluded: the agent began speaking "
            f"before the caller stopped."
        )
    return line + "\n"


def render_rates(runs: list[list[Verdict]]) -> str:
    """Per-scenario pass rate across repeated runs.

    Shown instead of a single pass/fail column once ``--repeat`` is above one,
    because "4/5" and "5/5" are different claims and collapsing them to "pass"
    throws away the distinction that matters.
    """
    lines = ["| Scenario | Passed | Rate | Failure modes seen |", "|---|:--:|--:|---|"]
    for index, name in enumerate(v.scenario for v in runs[0]):
        outcomes = [run[index] for run in runs]
        # A skipped scenario is excluded from its own denominator rather than
        # counted either way. Counting it as a pass claims a test that never
        # ran; counting it as a failure blames the agent for the mode it was
        # invoked in.
        ran = [v for v in outcomes if not v.skipped]
        if not ran:
            lines.append(f"| {name} | - | skipped | {outcomes[0].skipped} |")
            continue
        passes = sum(1 for v in ran if v.passed)
        modes = sorted({_failure_mode(v) for v in ran if not v.passed})
        lines.append(
            f"| {name} | {passes}/{len(ran)} | {passes / len(ran):.0%} | "
            f"{', '.join(modes) or '-'} |"
        )
    return "\n".join(lines)


def _failure_mode(verdict: Verdict) -> str:
    """Why this run failed, in one word, so repeats can be compared."""
    if verdict.skipped:
        return "skipped"
    if verdict.error:
        return "error"
    if verdict.missing_tools:
        return "tool not called"
    if verdict.forbidden_said:
        return "said forbidden"
    if verdict.leaked_accounts:
        return "leaked account"
    if verdict.archetype_found is False or verdict.gap_named is False:
        return "wrong account"
    if verdict.grounded is False:
        return "grounding"
    if verdict.commit_truncated is False:
        return "barge-in"
    if verdict.handled_well is False:
        return "conduct"
    return "gave in"


def print_transcripts(runs: list[list[Verdict]]) -> None:
    """The dialogue behind every failure.

    Without this the only evidence of a failure is the judge's one-line note,
    which is enough to guess from and not enough to diagnose from.
    """
    for run_index, run in enumerate(runs, start=1):
        for verdict in run:
            if verdict.passed:
                continue
            print(f"\n<details><summary>{verdict.scenario} — run {run_index}</summary>\n")
            print(f"_{_failure_mode(verdict)}: {verdict.notes}_\n")
            for who, text in verdict.transcript:
                print(f"**{who}:** {text}\n")
            for entry in verdict.tool_log:
                print(f"> `{entry}`\n")
            print("</details>")


SUMMARY_PATH = Path("src/voice_agent/server/static/eval-summary.json")


def _record_summary(
    runs: list[list[Verdict]], scenarios: list[Scenario], *, judge_model: str
) -> None:
    """Write the result where the dashboard can read it.

    Committed on purpose. The dashboard has no way to compute correctness for a
    live conversation, so the honest thing it can show is the last real run of
    this suite, dated, with the command that produced it.
    """
    executed = [[v for v in run if not v.skipped] for run in runs]
    total = sum(len(run) for run in executed)
    passed = sum(1 for run in executed for v in run if v.passed)
    rates = []
    for index in range(len(scenarios)):
        ran = [run[index] for run in runs if not run[index].skipped]
        if ran:
            rates.append(
                {
                    "scenario": ran[0].scenario,
                    "passed": sum(1 for v in ran if v.passed),
                    "of": len(ran),
                }
            )
    payload = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": len(runs),
        "passed": passed,
        "of": total,
        "model": judge_model,
        "command": f"uv run python -m evals --repeat {len(runs)}",
        "scenarios": rates,
    }
    with contextlib.suppress(OSError):
        SUMMARY_PATH.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals", description=__doc__)
    parser.add_argument(
        "--audio",
        action="store_true",
        help="run through the real pipeline (slower; exercises VAD and barge-in)",
    )
    parser.add_argument("--only", default="", help="substring match on scenario file")
    parser.add_argument(
        "--judge-model",
        default="",
        help="Anthropic model for the caller and judge (default: the configured LLM)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run the whole suite N times and report a pass rate (default 1)",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=1.0,
        help=(
            "fail the run if any scenario passes less often than this "
            "(default 1.0, i.e. every scenario must pass every time)"
        ),
    )
    parser.add_argument(
        "--transcripts",
        action="store_true",
        help="print the dialogue and tool calls behind each failure",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = Settings()
    db_path = seed(settings.db_path)
    scenarios: list[Scenario] = [
        s
        for s in load_all()
        if not args.only
        or args.only.lower() in s.slug
        or args.only.lower() in s.source.lower()
        or args.only.lower() in s.name.lower()
    ]
    if not scenarios:
        print(f"no scenarios matched {args.only!r}")
        return 2

    mode = "audio" if args.audio else "text"
    print(f"# Evaluation — {platform.node()} ({mode} mode)")
    print()
    from evals import ground_truth

    print(f"- {len(scenarios)} scenarios, profile {settings.profile}")
    print(f"- {ground_truth.describe(db_path, settings.rep_name)}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        providers = build_providers(settings, client)
        judge_llm: LLM = providers.llm
        if args.judge_model:
            from voice_agent.providers.llm_anthropic import AnthropicLLM

            if settings.anthropic_api_key is None:
                print("VA_ANTHROPIC_API_KEY is required for --judge-model")
                return 2
            judge_llm = AnthropicLLM(
                settings.anthropic_api_key.get_secret_value(),
                model=args.judge_model,
                max_tokens=400,
            )
        print(f"- agent {providers.llm.name}, caller and judge {judge_llm.name}")
        print()

        if args.audio:
            await providers.warmup()

        runs: list[list[Verdict]] = []
        for attempt in range(max(1, args.repeat)):
            if args.repeat > 1:
                print(f"  --- run {attempt + 1} of {args.repeat}", flush=True)
            verdicts: list[Verdict] = []
            for scenario in scenarios:
                print(f"  running {scenario.name} ...", flush=True)
                if args.audio:
                    verdict = await run_audio(scenario, providers, settings, judge_llm, db_path)
                else:
                    verdict = await run_text(
                        scenario, providers.llm, judge_llm, db_path, settings.rep_name
                    )
                verdicts.append(verdict)
            runs.append(verdicts)

        await providers.aclose()
        if judge_llm is not providers.llm:
            await judge_llm.aclose()

    print()
    if len(runs) > 1:
        print(render_rates(runs))
    else:
        print(render(runs[0], audio=args.audio))
    every = [v for run in runs for v in run]
    print(summarise_latency(every, settings.budget))
    print(summarise_barge_in(every))

    # Notes from the most recent run; earlier ones are summarised by the rates
    # table, and repeating every note for every run buries the signal.
    for v in runs[-1]:
        if v.notes or not v.passed:
            detail = v.notes or ""
            if v.missing_tools:
                detail += f" missing tools: {', '.join(v.missing_tools)}."
            if v.forbidden_said:
                detail += f" said forbidden: {', '.join(v.forbidden_said)}."
            if v.faults_fired:
                detail += f" faults fired: {v.faults_fired}."
            print(f"- **{v.scenario}** — {detail.strip()}")

    if args.transcripts:
        print_transcripts(runs)

    _record_summary(runs, scenarios, judge_model=args.judge_model or "")

    rates = []
    for i in range(len(scenarios)):
        ran = [run[i] for run in runs if not run[i].skipped]
        # A scenario skipped in every run cannot fail the gate for not running.
        rates.append(sum(1 for v in ran if v.passed) / len(ran) if ran else 1.0)
    executed = sum(1 for run in runs for v in run if not v.skipped)
    total = sum(1 for run in runs for v in run if v.passed)
    print()
    if len(runs) > 1:
        print(
            f"{total}/{executed} scenario-runs passed "
            f"across {len(runs)} runs — worst scenario {min(rates):.0%}"
        )
    else:
        print(f"{total}/{executed} passed")
    return 0 if min(rates) >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
