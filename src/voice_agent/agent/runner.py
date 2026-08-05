"""What the pipeline talks to for a turn's worth of reasoning.

The pipeline should not know whether there is a tool-using graph behind it or a
bare model. Both satisfy :class:`TurnSource`, so the turn loop — clause cutting,
synthesis, playback, barge-in — is identical either way, and the agent layer is
a configuration choice rather than a fork in the code.

:class:`DirectTurnSource` is the no-graph implementation. It exists so the
repository still runs with only the ``local`` extra installed: LangGraph is an
optional dependency, and a missing optional dependency should cost you tools,
not the ability to hold a conversation.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from voice_agent.agent.memory import ConversationStore
from voice_agent.config import Settings
from voice_agent.providers.base import LLM, LlmDelta, Message

log = logging.getLogger(__name__)


@runtime_checkable
class TurnSource(Protocol):
    """Produces one turn of assistant speech, and records what was heard."""

    #: Named for logs and the health endpoint.
    name: str
    #: Non-empty when the agent is running without its tool graph.
    degraded_reason: str | None

    def stream(self, thread_id: str, user_text: str) -> AsyncIterator[LlmDelta]:
        """Stream the reply as deltas."""
        ...

    async def commit(self, thread_id: str, spoken_text: str) -> None:
        """Record what the caller actually heard.

        Called once playback ends or is cut short — never with the generated
        text, which after an interruption is more than was said.
        """
        ...

    async def history(self, thread_id: str) -> list[Message]:
        """The conversation so far, for the call summary."""
        ...


class DirectTurnSource:
    """A turn source with no agent layer: model in, speech out.

    Keeps a conversation window so the agent remembers the call, which is the
    minimum the previous implementation failed to do, but offers no tools.
    """

    name = "direct"

    def __init__(
        self,
        llm: LLM,
        store: ConversationStore,
        *,
        system: str,
        degraded_reason: str | None = None,
    ) -> None:
        self._llm = llm
        self._store = store
        self._system = system
        self.degraded_reason = degraded_reason

    async def stream(self, thread_id: str, user_text: str) -> AsyncIterator[LlmDelta]:
        messages = [
            *self._store.history(thread_id),
            Message(role="user", content=user_text),
        ]
        self._store.commit_turn(thread_id, user_text, "")
        async for delta in self._llm.stream(messages, system=self._system):
            yield delta

    async def commit(self, thread_id: str, spoken_text: str) -> None:
        self._store.commit_turn(thread_id, "", spoken_text)

    async def history(self, thread_id: str) -> list[Message]:
        return self._store.history(thread_id)


def build_turn_source(
    settings: Settings,
    llm: LLM,
    store: ConversationStore,
    *,
    system: str,
) -> TurnSource:
    """Choose the agent layer, degrading rather than failing.

    LangGraph is an optional extra. If it is absent, or the graph cannot be
    constructed, the call still works — with a conversation window and no
    tools. Refusing to answer the phone because a tool database is missing
    would be the wrong trade.
    """
    if not settings.agent_enabled:
        reason = "agent layer disabled by VA_AGENT_ENABLED=false; running without tools"
        log.warning("DEGRADED MODE: %s", reason)
        return DirectTurnSource(llm, store, system=system, degraded_reason=reason)

    try:
        from voice_agent.agent.graph import AgentRunner
        from voice_agent.agent.tools import build_toolbox
        from voice_agent.agent.tools.db import seed

        seed(settings.db_path)
        toolbox = build_toolbox(settings.db_path, llm, rep=settings.rep_name)
        runner = AgentRunner(llm, toolbox, system=system)
        log.info(
            "agent layer ready with %d tools: %s",
            len(toolbox.specs),
            ", ".join(spec.name for spec in toolbox.specs),
        )
        return runner
    except ImportError as exc:
        reason = "agent extra unavailable; install the agent extra to enable tools"
        log.error("DEGRADED MODE: %s (%s)", reason, exc)
    except Exception as exc:
        reason = f"agent layer failed to load; running without tools ({type(exc).__name__})"
        log.exception("DEGRADED MODE: %s", reason)
    return DirectTurnSource(llm, store, system=system, degraded_reason=reason)
