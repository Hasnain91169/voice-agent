"""An LLM playing the caller.

Scripted callers only ever test the happy path. A model on the other end of the
line will mishear, change its mind, repeat itself, and go quiet — and those are
the turns where a voice agent actually falls over. The persona in each scenario
decides how awkward it is willing to be.

The caller is deliberately kept ignorant of the agent's tools and database. It
knows who it is and what it wants, which is all a real caller knows.
"""

from __future__ import annotations

import logging

from evals.scenario import Scenario
from voice_agent.providers.base import LLM, Message, TextDelta

log = logging.getLogger(__name__)

#: The caller says this when it has what it came for. Cheaper and more reliable
#: than asking a judge whether the call is over.
#: What the caller is shown when the agent said nothing at all.
SILENCE = "(the line goes quiet)"

END_MARKER = "[END]"

_INSTRUCTIONS = """You are role-playing a person on a phone call to a supplier.

{persona}

What you want from this call:
{goal}

Rules:
- Reply with exactly what you would say next, and nothing else. No narration,
  no quotation marks, no stage directions.
- One or two sentences. People are brief on the phone.
- Stay in character. You do not know how the supplier's systems work.
- If the agent has answered you, or the call has clearly run its course, reply
  with exactly {end} and nothing else.
- If the agent says something that does not answer you, say so plainly.
- {silence} means you heard nothing at all. React the way anyone does to a
  quiet line: check they are still there, or repeat yourself.
- The agent may be cut off mid-word, so its last turn can be a fragment or a
  single word. That is a person being interrupted, not a broken script. React
  the way someone does on a phone call - carry on with what you were saying.
  Never comment on the exercise, never ask anyone to start, never describe
  what you are about to do. You are on a call, not in a rehearsal.
"""


class SimulatedCaller:
    """Generates the caller's side of a scenario."""

    def __init__(self, llm: LLM, scenario: Scenario) -> None:
        self._llm = llm
        self._scenario = scenario
        self._history: list[Message] = []
        self.turns = 0

    @property
    def opening(self) -> str:
        return self._scenario.opening

    def record_agent(self, said: str) -> None:
        """Note what the agent said, from the caller's point of view.

        Roles are inverted relative to the agent's own history: to the caller,
        the agent is the other party.

        Silence is recorded as silence rather than skipped. In audio mode a turn
        can legitimately produce no committed text — a blanked transcript plays
        a cached clarifier that never enters history — and dropping it left the
        caller with nothing to answer, which ended the call with a 400 from the
        API instead of the behaviour under test. A person on a quiet line says
        "hello?"; so should the simulation.
        """
        self._history.append(Message(role="user", content=said.strip() or SILENCE))

    def record_self(self, said: str) -> None:
        if said.strip():
            self._history.append(Message(role="assistant", content=said.strip()))

    async def next_turn(self) -> str | None:
        """The caller's next line, or ``None`` if they are done."""
        self.turns += 1
        if self.turns > self._scenario.max_turns:
            return None

        system = _INSTRUCTIONS.format(
            persona=self._scenario.persona,
            goal=self._scenario.goal,
            end=END_MARKER,
            silence=SILENCE,
        )
        # An empty history is not a request the API will accept, and the
        # caller has to open the call somehow.
        history = self._history or [Message(role="user", content=SILENCE)]
        parts: list[str] = []
        try:
            async for delta in self._llm.stream(history, system=system):
                if isinstance(delta, TextDelta):
                    parts.append(delta.text)
        except Exception as exc:
            log.warning("simulated caller failed: %s", exc)
            return None

        line = " ".join("".join(parts).split()).strip().strip('"')
        if not line or END_MARKER in line.upper():
            return None
        return line
