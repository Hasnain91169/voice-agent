"""Compare LLMs on the two things a voice agent needs from one.

    uv run python -m bench.models

**Latency**, because time-to-first-token lands directly in the first-audio
budget and nothing else the model does can make up for it. And **tool use**,
because an agent that answers fluently without looking anything up is worse
than useless on a support line — it is confidently wrong.

Both matter, and they trade against each other: the models that reason well
enough to chain a lookup into an answer are generally the slower ones. This
harness measures the trade rather than assuming it.
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from bench.stats import Measurement
from voice_agent.agent.graph import AgentRunner
from voice_agent.agent.prompts import SYSTEM_PROMPT
from voice_agent.agent.tools import Toolbox, build_toolbox
from voice_agent.agent.tools.db import seed
from voice_agent.config import Settings
from voice_agent.providers.base import LLM, TextDelta, ToolCall

#: A short exchange, representative of a turn the caller waits through.
LATENCY_PROMPT = "Hi, can you check my most recent order?"

#: The capability scenario. Each turn names the tool the model ought to reach
#: for and a fact that can only come from having actually called it.
SCENARIO: list[tuple[str, str, str]] = [
    (
        "Hi, this is Crestline Electrical Wholesale calling.",
        "lookup_customer",
        "",
    ),
    (
        "What were my last two orders?",
        "get_order_history",
        "order|june|shipped",
    ),
    (
        "How many customers do you have in Yorkshire?",
        "query_business_data",
        # Alternatives, because the system prompt asks for numbers written the
        # way they are spoken. Checking for the digit alone marks a correct
        # spoken answer wrong, which is a fault in the harness rather than the
        # model.
        "2|two",
    ),
]


@dataclass
class Result:
    label: str
    ttft: Measurement
    total: Measurement
    tools_expected: int = 0
    tools_called: int = 0
    facts_right: int = 0
    facts_checked: int = 0
    transcript: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def tool_rate(self) -> str:
        if not self.tools_expected:
            return "-"
        return f"{self.tools_called}/{self.tools_expected}"

    @property
    def fact_rate(self) -> str:
        if not self.facts_checked:
            return "-"
        return f"{self.facts_right}/{self.facts_checked}"


async def measure_latency(llm: LLM, result: Result, runs: int) -> None:
    """Time to first token, warm."""
    from voice_agent.providers.base import Message

    prompt = [Message(role="user", content=LATENCY_PROMPT)]
    await llm.warmup()  # cold first calls measure model load, not inference

    for _ in range(runs):
        start = time.perf_counter()
        first: float | None = None
        try:
            async for delta in llm.stream(prompt, system=SYSTEM_PROMPT):
                if isinstance(delta, TextDelta) and first is None:
                    first = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            return
        if first is not None:
            result.ttft.add(first)
        result.total.add((time.perf_counter() - start) * 1000.0)


async def measure_capability(llm: LLM, toolbox: Toolbox, result: Result) -> None:
    """Run the scenario and score tool use and factual grounding."""
    runner = AgentRunner(llm, toolbox, system=SYSTEM_PROMPT)
    thread = f"bench-{result.label}"

    for turn, (utterance, expected_tool, expected_fact) in enumerate(SCENARIO):
        parts: list[str] = []
        called: set[str] = set()
        try:
            async for delta in runner.stream(thread, utterance):
                if isinstance(delta, TextDelta):
                    parts.append(delta.text)
                elif isinstance(delta, ToolCall):
                    called.add(delta.name)
        except Exception as exc:
            result.error = f"turn {turn}: {type(exc).__name__}: {exc}"
            return

        reply = "".join(parts).strip()
        await runner.commit(thread, reply)
        result.transcript.append(f"{utterance}  ->  {reply[:110]}")

        result.tools_expected += 1
        if expected_tool in called:
            result.tools_called += 1
        if expected_fact:
            result.facts_checked += 1
            lowered = reply.lower()
            if any(option in lowered for option in expected_fact.lower().split("|")):
                result.facts_right += 1


def render(results: list[Result]) -> str:
    lines = [
        "| Model | TTFT p50 | TTFT p95 | Total p50 | Tools | Facts |",
        "|---|---:|---:|---:|:--:|:--:|",
    ]
    for r in results:
        if not r.ttft.ok:
            lines.append(f"| {r.label} | — | — | — | — | _{r.error or 'no data'}_ |")
            continue
        # An error during the capability run must still surface: latency
        # succeeding is not evidence the scenario did.
        note = f" _{r.error[:60]}_" if r.error else ""
        lines.append(
            f"| {r.label} | {r.ttft.p50:.0f}ms | {r.ttft.p95:.0f}ms "
            f"| {r.total.p50:.0f}ms | {r.tool_rate} | {r.fact_rate}{note} |"
        )
    return "\n".join(lines)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.models", description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="latency runs per model")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["ollama", "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"],
        help="'ollama' for the local model, otherwise Anthropic model ids",
    )
    parser.add_argument("--skip-capability", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = Settings()
    db_path = seed(Path("data/wholesale.db"))

    print(f"# Model comparison — {platform.node()}")
    print()
    print(f"- {args.runs} latency runs per model, after a discarded warm-up")
    print(f"- capability scenario: {len(SCENARIO)} turns with tools available")
    print()

    results: list[Result] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
        for name in args.models:
            label = settings.ollama_model if name == "ollama" else name
            result = Result(
                label=label,
                ttft=Measurement("llm", label),
                total=Measurement("llm", label),
            )
            try:
                llm = _build(name, settings, client)
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                results.append(result)
                continue

            print(f"  measuring {label} ...", flush=True)
            await measure_latency(llm, result, args.runs)
            if not args.skip_capability and not result.error:
                toolbox = build_toolbox(db_path, llm)
                await measure_capability(llm, toolbox, result)
            await llm.aclose()
            results.append(result)

    print()
    print(render(results))
    print()
    print(
        "Tools = turns where the expected tool was called. "
        "Facts = answers containing a value only obtainable from the database."
    )
    print()
    for r in results:
        if r.transcript:
            print(f"### {r.label}")
            for line in r.transcript:
                print(f"- {line}")
            print()
    return 0


def _build(name: str, settings: Settings, client: httpx.AsyncClient) -> LLM:
    if name == "ollama":
        from voice_agent.providers.llm_ollama import OllamaLLM

        return OllamaLLM.from_settings(settings, client)

    from voice_agent.providers.llm_anthropic import AnthropicLLM

    if settings.anthropic_api_key is None:
        raise RuntimeError("VA_ANTHROPIC_API_KEY is not set")
    return AnthropicLLM(
        settings.anthropic_api_key.get_secret_value(),
        model=name,
        max_tokens=settings.llm_max_tokens,
        effort=settings.anthropic_effort,
        thinking=settings.anthropic_thinking,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
