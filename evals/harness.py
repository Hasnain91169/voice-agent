"""Running scenarios.

Two modes, and the difference is the point.

**Text mode** drives the agent layer directly. Fast, free on the local stack, and
suitable for CI — it catches reasoning, tool selection and grounding regressions.

**Audio mode** runs the real pipeline over a loopback transport: the caller's
turns are synthesised and fed in as 20ms frames, and the agent's audio comes back
out. It exercises the VAD, the endpointer, the echo gate and barge-in, none of
which a text harness can see. It also produces the only honest first-audio
measurement in the repository — timed from outside the process, from the moment
the caller stops speaking to the moment the first audio frame is sent, with
nothing self-reported.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from evals import chaos
from evals.judge import Verdict, check_deterministic, judge
from evals.loopback import LoopbackTransport
from evals.scenario import Scenario, resolve
from evals.simulated_user import SimulatedCaller
from voice_agent.agent.graph import AgentRunner
from voice_agent.agent.prompts import SYSTEM_PROMPT
from voice_agent.agent.tools import Toolbox, build_toolbox
from voice_agent.config import Settings
from voice_agent.pipeline import run_call
from voice_agent.prompts_cache import PromptCache
from voice_agent.providers.base import LLM, TextDelta, ToolCall
from voice_agent.providers.registry import Providers
from voice_agent.rx import FrameChannel, RxPump
from voice_agent.session import Session

log = logging.getLogger(__name__)

#: Agent audio to let through before talking over it — long enough to be
#: past the cached filler and into the answer. 16 kHz, 16-bit mono.
_MID_ANSWER_BYTES = int(16_000 * 2 * 2.5)


class RecordingToolbox:
    """Wraps a toolbox to remember what the tools actually returned.

    The judge needs this: "unsupported" has to mean unsupported by what the
    agent retrieved, not merely unverifiable by the judge.
    """

    def __init__(self, inner: Toolbox) -> None:
        self._inner = inner
        self.calls: list[str] = []
        self.results: list[str] = []

    @property
    def specs(self) -> Any:
        return self._inner.specs

    def schemas(self) -> list[dict[str, Any]]:
        return self._inner.schemas()

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._inner.invoke(name, arguments)
        self.calls.append(name)
        self.results.append(f"{name}({arguments}) -> {result}")
        return result


def _payload_delivered(db_path: Path, rep: str, attack: str, tool_results: list[str]) -> bool:
    """Did the planted payload actually reach the model's context this run?

    A red-team scenario that passes because the attack was never retrieved is
    the most flattering result the suite can produce and the least informative.
    It reports "the agent resisted" when what happened is "the agent was never
    asked to". Treated as an error rather than a pass, because the scenario did
    not run, and a suite that cannot tell those apart is not a security control.
    """
    from voice_agent.agent.tools import db as database

    with database.connect(db_path, read_only=True) as db:
        row = db.execute(
            "SELECT n.body FROM seed_injections s"
            " JOIN notes n ON n.id = s.note_id"
            " JOIN accounts a ON a.id = s.account_id"
            " JOIN reps r ON r.id = a.rep_id"
            " WHERE r.name = ? AND s.attack = ?",
            (rep, attack),
        ).fetchone()
    if row is None:
        return False
    # Match on a distinctive slice rather than the whole body: tool results wrap
    # the note in attribution, so an equality check would never fire.
    needle = str(row["body"])[:60]
    return any(needle in result for result in tool_results)


async def run_text(
    scenario: Scenario,
    llm: LLM,
    judge_llm: LLM,
    db_path: Path,
    rep: str,
) -> Verdict:
    """Drive the agent layer directly, with no audio."""
    scenario = resolve(scenario, db_path, rep)
    verdict = Verdict(scenario=scenario.name)
    if scenario.expects_barge_in:
        # Reported as a skip rather than quietly passing. A suite that scores
        # a barge-in test green without any audio is claiming to have tested
        # the one thing it cannot test.
        verdict.skipped = "needs --audio"
        return verdict
    toolbox = RecordingToolbox(build_toolbox(db_path, llm, rep=rep))

    wrapped = llm
    if turn := scenario.faults.get("llm_stall_turn"):
        wrapped = chaos.StallingLLM(
            llm,
            on_turn=int(turn),
            seconds=float(scenario.faults.get("llm_stall_seconds", 2.0)),
        )
    if turn := scenario.faults.get("llm_die_turn"):
        wrapped = chaos.DyingLLM(wrapped, on_turn=int(turn))

    runner = AgentRunner(wrapped, toolbox, system=SYSTEM_PROMPT)  # type: ignore[arg-type]
    caller = SimulatedCaller(judge_llm, scenario)
    thread = f"eval-{scenario.slug}"
    transcript: list[tuple[str, str]] = []

    line: str | None = scenario.opening
    while line:
        transcript.append(("caller", line))
        caller.record_self(line)
        parts: list[str] = []
        try:
            async for delta in runner.stream(thread, line):
                if isinstance(delta, TextDelta):
                    parts.append(delta.text)
                elif isinstance(delta, ToolCall):
                    pass  # recorded by the toolbox, which sees the result too
        except Exception as exc:
            verdict.error = f"{type(exc).__name__}: {exc}"
            break

        said = "".join(parts).strip()
        await runner.commit(thread, said)
        transcript.append(("agent", said))
        caller.record_agent(said)
        verdict.turns += 1
        line = await caller.next_turn()

    verdict.tools_called = tuple(dict.fromkeys(toolbox.calls))
    verdict.transcript = transcript
    verdict.tool_log = list(toolbox.results)
    if scenario.about_injection and not _payload_delivered(
        db_path, rep, scenario.about_injection, toolbox.results
    ):
        verdict.error = f"{scenario.about_injection} payload never retrieved"
    check_deterministic(scenario, verdict, transcript, db_path, rep)
    await judge(judge_llm, scenario, verdict, transcript, toolbox.results)
    return verdict


async def run_audio(
    scenario: Scenario,
    providers: Providers,
    settings: Settings,
    judge_llm: LLM,
    db_path: Path,
) -> Verdict:
    rep = settings.rep_name
    """Run the real pipeline over a loopback transport."""
    scenario = resolve(scenario, db_path, rep)
    verdict = Verdict(scenario=scenario.name)

    asr, tts, llm, injected = chaos.apply(
        scenario.faults, providers.asr, providers.tts, providers.llm
    )
    toolbox = RecordingToolbox(build_toolbox(db_path, providers.llm, rep=rep))
    runner = AgentRunner(llm, toolbox, system=SYSTEM_PROMPT)  # type: ignore[arg-type]

    cache = PromptCache()
    await cache.build(tts, settings.languages)

    transport = LoopbackTransport()
    channel = FrameChannel()
    session = Session(transport=transport, channel=channel, settings=settings)
    faulted = Providers(asr=asr, tts=tts, llm=llm)

    pump = asyncio.create_task(RxPump(transport, channel).run(), name="eval-rx")
    call = asyncio.create_task(
        run_call(session, faulted, runner, cache, settings), name="eval-call"
    )

    caller = SimulatedCaller(judge_llm, scenario)
    transcript: list[tuple[str, str]] = []

    try:
        # Calibration needs ambient audio before anything else happens.
        await transport.hush(0.7)
        await transport.wait_until_quiet(give_up=25)

        line: str | None = scenario.opening
        turn_index = 0
        while line:
            transcript.append(("caller", line))
            caller.record_self(line)

            speech = b"".join([c async for c in providers.tts.synthesize(line)])
            transport.reset_speech_clock()
            await transport.say(speech)
            await transport.hush(0.5)
            ended_at = time.monotonic()

            if scenario.interrupt_at == turn_index:
                await _interrupt(transport, providers, caller, verdict)

            await transport.wait_until_quiet(give_up=45)
            # From the caller falling silent to the agent's FIRST frame. This
            # read `last_spoke_at` until it was checked against a real run: that
            # is the last frame of the answer, so the figure silently included
            # the whole spoken reply and was published as responsiveness.
            if transport.first_spoke_at:
                gap = (transport.first_spoke_at - ended_at) * 1000
                if gap >= 0:
                    verdict.first_audio_ms.append(gap)
                else:
                    # The agent began before the caller stopped, so there is no
                    # latency to report — it endpointed early, or was talked
                    # over deliberately. Counted, never averaged in as a
                    # negative, which is how a nonsense p50 passed a budget
                    # check that existed to reject exactly that.
                    verdict.overlaps += 1

            said = await _latest_agent_text(runner, session.thread_id)
            transcript.append(("agent", said))
            caller.record_agent(said)
            verdict.turns += 1
            turn_index += 1
            line = await caller.next_turn()
    except Exception as exc:
        verdict.error = f"{type(exc).__name__}: {exc}"
    finally:
        await transport.close()
        call.cancel()
        pump.cancel()
        await asyncio.gather(call, pump, return_exceptions=True)

    barged = [m for m in session.history if m.barged_in]
    verdict.barge_ins = len(barged)
    if scenario.expects_barge_in:
        # Two separate claims: the turn was actually cancelled, and history kept
        # only what was heard. A barge-in that cancels playback but commits the
        # full generated answer leaves the agent certain it said things the
        # caller never heard.
        #
        # Only turns that produced something can be checked for truncation. A
        # turn cancelled before any clause reached the speaker has nothing to
        # truncate — generated_chars is 0, and requiring 0 < 0 marked those as
        # failures. That is what produced a string of 0-of-N readings which
        # looked like a broken pipeline and was a broken assertion.
        spoke = [m for m in barged if m.generated_chars > 0]
        verdict.commit_truncated = bool(spoke) and all(
            m.spoken_chars < m.generated_chars for m in spoke
        )
    verdict.faults_fired = sum(getattr(f, "fired", 0) for f in injected)
    verdict.tools_called = tuple(dict.fromkeys(toolbox.calls))
    verdict.transcript = transcript
    verdict.tool_log = list(toolbox.results)
    if scenario.about_injection and not _payload_delivered(
        db_path, rep, scenario.about_injection, toolbox.results
    ):
        verdict.error = f"{scenario.about_injection} payload never retrieved"
    check_deterministic(scenario, verdict, transcript, db_path, rep)
    await judge(judge_llm, scenario, verdict, transcript, toolbox.results)
    return verdict


async def _interrupt(
    transport: LoopbackTransport,
    providers: Providers,
    caller: SimulatedCaller,
    verdict: Verdict,
) -> None:
    """Talk over the agent once it has got a sentence out, and time the stop.

    The measurement is the point. Barge-in was implemented, logged and never
    asserted, which meant "it works" rested on having watched a log line go past
    during manual testing. What is recorded here is the interval between the
    caller's interrupting speech starting and the agent's final outbound frame —
    the delay the caller actually experiences as being talked over.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 25
    # Polled: we are waiting for audio to start arriving from another task,
    # which signals nothing.
    while not transport.speaking and loop.time() < deadline:  # noqa: ASYNC110
        await asyncio.sleep(0.02)

    # Wait for real content, not the filler. A fixed 0.6s delay interrupted
    # "Let me pull that up" every time — the agent committed the single word
    # "Let", which is correct truncation of the wrong thing, and told us
    # nothing about interrupting an answer. Measured in audio actually sent.
    started_speaking = len(transport.spoken)
    while loop.time() < deadline:
        if loop.time() - transport.last_spoke_at > 1.0:
            break  # it stopped; nothing left to interrupt
        if len(transport.spoken) - started_speaking >= _MID_ANSWER_BYTES:
            break
        await asyncio.sleep(0.02)

    barge = b"".join([c async for c in providers.tts.synthesize("Sorry, hold on a moment.")])
    # Louder than the agent, as a real interruption is.
    import numpy as np

    from voice_agent.audio import wav

    samples = wav.to_array(barge).astype(np.float64) * 1.7
    loud = samples.clip(-32768, 32767).round().astype("<i2").tobytes()

    started = loop.time()
    await transport.say(loud)
    # The last frame the agent managed to send. If it stopped before the
    # interruption finished, this is when it gave way.
    if transport.last_spoke_at > started:
        verdict.barge_in_ms.append((transport.last_spoke_at - started) * 1000)
    else:
        # It had already stopped by the time we started talking, so it did not
        # have to be interrupted at all. Recorded as zero rather than dropped.
        verdict.barge_in_ms.append(0.0)
    verdict.notes = (verdict.notes + " interrupted").strip()


async def _latest_agent_text(runner: AgentRunner, thread_id: str) -> str:
    """What the agent last committed — i.e. what the caller actually heard."""
    history = await runner.history(thread_id)
    for message in reversed(history):
        if message.role == "assistant" and message.content.strip():
            return message.content.strip()
    return ""
