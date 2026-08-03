"""The agent graph.

``ingest -> agent -> (tools -> agent)* -> end``, with LangGraph owning the
conversation state and checkpointing it per thread.

**LangGraph runs the reasoning; it never touches audio.** The graph streams text
deltas out through a custom stream channel, and :mod:`voice_agent.pipeline`
consumes that stream exactly as it consumed the raw LLM stream before — cutting
clauses, synthesising, pacing playback, and cancelling the lot on a barge-in.
Putting the framework inside the 20ms frame loop would make interruption a
matter of hoping the graph notices; keeping it out means a barge-in cancels an
async iterator, which is a solved problem.

One subtlety drives the shape of the ``agent`` node. The assistant's final reply
is deliberately **not** appended to state by the graph. The caller may interrupt
partway through, so what belongs in history is what was actually spoken, which is
only known once playback finishes. The pipeline calls :meth:`AgentRunner.commit`
after the turn with that text. Intermediate assistant turns that carry tool calls
*are* appended, because the tool loop cannot proceed without them.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from voice_agent.agent.tools import Toolbox
from voice_agent.providers.base import LLM, LlmDelta, Message, TextDelta, ToolCall

log = logging.getLogger(__name__)

#: Tool rounds allowed within one turn. A model that has not answered after a
#: few lookups is looping, and a caller is on the line: better a partial answer
#: than a silent minute.
MAX_TOOL_ROUNDS = 3


class AgentState(TypedDict, total=False):
    """Everything the agent reasons over. The authoritative conversation record."""

    # Stored as plain dicts, not as Message instances. The checkpointer
    # serialises state with msgpack, which does not know custom dataclasses;
    # LangGraph warns today and will refuse in a future version. Dicts also
    # keep a checkpoint readable by anything that opens the database.
    messages: Annotated[list[dict[str, Any]], operator.add]
    #: Tool calls awaiting execution this round.
    pending: list[ToolCall]
    #: Rounds of tool use spent on the current turn.
    rounds: int
    #: The most recent full generation, for diagnostics. Not what gets
    #: committed — see the module docstring.
    generated: str


class AgentRunner:
    """Runs the graph for one turn and streams its text out."""

    name = "langgraph"

    def __init__(
        self,
        llm: LLM,
        toolbox: Toolbox,
        *,
        system: str,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self._llm = llm
        self._toolbox = toolbox
        self._system = system
        self._max_rounds = max_tool_rounds
        self._checkpointer = MemorySaver()
        self._graph = self._build()

    def drain_tool_results(self) -> list[str]:
        """Everything the tools returned since the last drain.

        Exists for the grounding trace, which needs the text the model was
        actually handed rather than a summary of it. Draining rather than
        reading keeps it to one turn — a figure from two turns ago is not
        evidence for what was just said.
        """
        results = list(self._toolbox.results)
        self._toolbox.results.clear()
        return results

    # ------------------------------------------------------------------ graph

    def _build(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self._route, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")
        return graph.compile(checkpointer=self._checkpointer)

    def _route(self, state: AgentState) -> str:
        if not state.get("pending"):
            return "end"
        if state.get("rounds", 0) >= self._max_rounds:
            log.warning("tool round limit reached; answering with what we have")
            return "end"
        return "tools"

    async def _agent_node(self, state: AgentState) -> dict[str, Any]:
        """Stream a reply, emitting text as it arrives and collecting tool calls."""
        writer = get_stream_writer()
        messages = [_to_message(entry) for entry in state.get("messages", [])]
        calls: list[ToolCall] = []
        text: list[str] = []

        async for delta in self._llm.stream(
            messages, system=self._system, tools=self._toolbox.schemas()
        ):
            if isinstance(delta, TextDelta):
                text.append(delta.text)
                # Straight out to the pipeline, which speaks it. This is the
                # only path by which the graph produces audible output.
                writer(delta)
            elif isinstance(delta, ToolCall):
                calls.append(delta)
                # Surfaced for metrics and logging only. The pipeline skips
                # these rather than speaking them.
                writer(delta)

        generated = "".join(text)
        update: dict[str, Any] = {"pending": calls, "generated": generated}

        if calls:
            # Needed for the tool loop: the tool results that follow have to
            # answer a recorded assistant turn. A final, tool-free reply is
            # committed later by the pipeline with what the caller actually
            # heard.
            update["messages"] = [
                _from_message(
                    Message(
                        role="assistant",
                        content=generated,
                        # The calls must travel with the turn. Without them the
                        # tool results that follow answer nothing, and the
                        # Messages API rejects the next request outright:
                        # "each tool_result block must have a corresponding
                        # tool_use block in the previous message".
                        tool_calls=tuple(calls),
                    )
                )
            ]
            update["rounds"] = state.get("rounds", 0) + 1
        return update

    async def _tools_node(self, state: AgentState) -> dict[str, Any]:
        """Execute every pending tool and feed the results back."""
        results: list[dict[str, Any]] = []
        for call in state.get("pending", []):
            output = await self._toolbox.invoke(call.name, call.arguments)
            results.append(
                _from_message(
                    Message(
                        role="tool",
                        content=output,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
            )
        return {"messages": results, "pending": []}

    # ------------------------------------------------------------------- api

    async def stream(self, thread_id: str, user_text: str) -> AsyncIterator[LlmDelta]:
        """Run one turn, yielding text deltas as the model produces them."""
        config = {"configurable": {"thread_id": thread_id}}
        inputs: AgentState = {
            "messages": [_from_message(Message(role="user", content=user_text))],
            "pending": [],
            "rounds": 0,
        }
        async for chunk in self._graph.astream(inputs, config=config, stream_mode="custom"):
            if isinstance(chunk, TextDelta | ToolCall):
                yield chunk

    async def commit(self, thread_id: str, spoken_text: str) -> None:
        """Record what the caller actually heard as the assistant's turn.

        The single crossing point back into conversation state, called once
        playback has finished or been cut short. Committing the generated text
        instead would leave the agent believing it had said things the caller
        never heard.
        """
        if not spoken_text.strip():
            return
        config = {"configurable": {"thread_id": thread_id}}
        await self._graph.aupdate_state(
            config,
            {"messages": [_from_message(Message(role="assistant", content=spoken_text.strip()))]},
        )

    async def history(self, thread_id: str) -> list[Message]:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self._graph.aget_state(config)
        entries: Sequence[dict[str, Any]] = (snapshot.values or {}).get("messages", [])
        return [_to_message(entry) for entry in entries]


def _from_message(message: Message) -> dict[str, Any]:
    """Flatten to the plain dict the checkpointer can serialise."""
    return {
        "role": message.role,
        "content": message.content,
        "tool_call_id": message.tool_call_id,
        "name": message.name,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ],
    }


def _to_message(entry: dict[str, Any]) -> Message:
    return Message(
        role=entry["role"],
        content=entry.get("content", ""),
        tool_call_id=entry.get("tool_call_id"),
        name=entry.get("name"),
        tool_calls=tuple(
            ToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments", {}))
            for c in entry.get("tool_calls", [])
        ),
    )
