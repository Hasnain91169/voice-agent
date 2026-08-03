"""Conversation state.

The authoritative store for everything the agent reasons over. The pipeline
never keeps message history of its own — it holds a ``thread_id`` and calls
:meth:`ConversationStore.commit_turn` once per turn.

The gateway this replaces had no memory at all: it built a fresh single-turn
prompt for every reply. The effect is visible in its own committed transcripts,
where the agent asks the same caller to confirm his name four times in a row.

Phase 3 replaces this implementation with a LangGraph checkpointer. The
interface is the boundary that makes that a swap rather than a rewrite, which is
why the crossing point is deliberately narrow: one call, one direction, at the
end of a turn.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from voice_agent.providers.base import Message

log = logging.getLogger(__name__)

#: Turns retained per thread. A phone call rarely runs long enough to need
#: compaction, and a bounded window keeps prompt size — and therefore
#: time-to-first-token — predictable.
DEFAULT_WINDOW_TURNS = 12


@dataclass
class Fact:
    """Something established during the call, worth surviving the window."""

    key: str
    value: str


@dataclass
class Thread:
    messages: deque[Message]
    facts: dict[str, str] = field(default_factory=dict)


class ConversationStore:
    """In-memory conversation history, keyed by thread."""

    def __init__(self, window_turns: int = DEFAULT_WINDOW_TURNS) -> None:
        # Two messages per turn, so the deque holds twice the turn window.
        self._window = window_turns * 2
        self._threads: dict[str, Thread] = defaultdict(
            lambda: Thread(messages=deque(maxlen=self._window))
        )

    def history(self, thread_id: str) -> list[Message]:
        return list(self._threads[thread_id].messages)

    def commit_turn(self, thread_id: str, user_text: str, spoken_text: str) -> None:
        """Record one exchange.

        ``spoken_text`` is what the caller actually heard, not what was
        generated. After an interruption those differ, and committing the
        generated text would leave the agent reasoning from a false record of
        its own side of the conversation for the rest of the call.
        """
        thread = self._threads[thread_id]
        if user_text.strip():
            thread.messages.append(Message(role="user", content=user_text.strip()))
        if spoken_text.strip():
            thread.messages.append(Message(role="assistant", content=spoken_text.strip()))

    def remember(self, thread_id: str, key: str, value: str) -> None:
        """Pin a fact so it outlives the rolling message window."""
        self._threads[thread_id].facts[key] = value

    def facts(self, thread_id: str) -> dict[str, str]:
        return dict(self._threads[thread_id].facts)

    def transcript(self, thread_id: str) -> list[Message]:
        """Everything retained for this thread, for the call summary."""
        return self.history(thread_id)

    def forget(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)
