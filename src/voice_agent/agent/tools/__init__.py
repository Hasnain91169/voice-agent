"""The tool suite offered to the agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from voice_agent.agent.tools.base import ToolError, ToolSpec
from voice_agent.providers.base import LLM

log = logging.getLogger(__name__)

__all__ = ["ToolError", "ToolSpec", "Toolbox", "build_toolbox"]


class Toolbox:
    """Named tools, with dispatch."""

    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        #: What the tools returned this turn. Read by the grounding trace, which
        #: needs the text the model was actually given rather than a summary of
        #: it, and cleared at the start of each turn by the pipeline.
        self.results: list[str] = []

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def schemas(self) -> list[dict[str, Any]]:
        """Provider-neutral schemas, converted by each LLM adapter."""
        return [spec.as_dict() for spec in self._specs.values()]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool and return a result the model can read out.

        Never raises. A tool that fails returns a sentence explaining what went
        wrong, because the agent's best move is almost always to tell the caller
        and offer something else — not to abandon the turn. An exception raised
        here would take the whole turn down with it, and the caller would hear
        the error line instead of an answer.
        """
        spec = self._specs.get(name)
        if spec is None:
            log.warning("model called unknown tool %r", name)
            self.results.append(_r := f"There is no tool called {name}.")
            return _r

        try:
            result = await spec.handler(**arguments)
        except TypeError as exc:
            # Wrong or missing arguments: say precisely what was wrong, so the
            # model can retry with the right ones rather than giving up.
            log.warning("bad arguments for %s: %s", name, exc)
            self.results.append(_r := f"That call to {name} had the wrong arguments: {exc}.")
            return _r
        except ToolError as exc:
            self.results.append(_r := str(exc))
            return _r
        except Exception as exc:
            log.exception("tool %s failed", name)
            self.results.append(_r := f"The {name} lookup failed: {type(exc).__name__}.")
            return _r

        log.info("tool %s(%s) -> %s", name, json.dumps(arguments)[:120], result[:160])
        self.results.append(result)
        return result


def build_toolbox(db_path: Path, llm: LLM, *, rep: str) -> Toolbox:
    """Assemble every tool, scoped to one rep.

    ``llm`` is needed by the natural-language query tool, which generates SQL.
    """
    from voice_agent.agent.tools import sales, sql

    return Toolbox([*sales.build(db_path, rep), *sql.build(db_path, llm)])
