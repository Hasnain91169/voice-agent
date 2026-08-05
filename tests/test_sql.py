"""The natural-language SQL tool must stay inside the current rep's patch."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from voice_agent.agent.tools import db as database
from voice_agent.agent.tools import sql
from voice_agent.providers.base import LlmDelta, Message, TextDelta


class StubLLM:
    name = "stub"

    def __init__(self, statement: str) -> None:
        self.statement = statement

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LlmDelta]:
        yield TextDelta(text=self.statement)


def _account_names(db_path: Path, rep: str) -> tuple[str, str]:
    with database.connect(db_path, read_only=True) as connection:
        mine = connection.execute(
            "SELECT a.name FROM accounts a JOIN reps r ON r.id = a.rep_id "
            "WHERE r.name = ? LIMIT 1",
            (rep,),
        ).fetchone()
        other = connection.execute(
            "SELECT a.name FROM accounts a JOIN reps r ON r.id = a.rep_id "
            "WHERE r.name != ? LIMIT 1",
            (rep,),
        ).fetchone()
    assert mine is not None and other is not None
    return str(mine["name"]), str(other["name"])


async def test_generated_sql_cannot_return_another_reps_account(tmp_path: Path) -> None:
    rep = "Dani Brooks"
    db_path = database.seed(tmp_path / "wholesale.db")
    _, other_account = _account_names(db_path, rep)
    tool = sql.build(
        db_path,
        StubLLM(f"SELECT name FROM my_accounts WHERE name = '{other_account}'"),
        rep=rep,
    )[0]

    result = await tool.handler(question="find this account")

    assert result == "That returned no results."


async def test_generated_sql_cannot_use_global_base_tables(tmp_path: Path) -> None:
    db_path = database.seed(tmp_path / "wholesale.db")
    tool = sql.build(db_path, StubLLM("SELECT name FROM accounts"), rep="Dani Brooks")[0]

    result = await tool.handler(question="list accounts")

    assert "unknown or unreadable table(s): accounts" in result
