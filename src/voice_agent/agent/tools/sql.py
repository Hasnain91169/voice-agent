"""Natural-language querying over the business data.

The interesting tool in the suite, and the dangerous one: the model writes SQL
and the server runs it. Three independent guards, because any one of them alone
is a bad bet:

1. **The connection cannot write.** Opened with SQLite's ``mode=ro`` URI, so a
   DROP or UPDATE fails at the engine regardless of what got past inspection.
   Pattern-matching statements is always one clever payload from being wrong;
   a connection that physically cannot write is not.
2. **One statement, SELECT only, allowlisted tables.** Catches the obvious
   cases early and with a comprehensible error the model can recover from.
3. **A row cap and a timeout.** A cross join over the order items would
   otherwise stall a live phone call.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from pathlib import Path

from voice_agent.agent.tools.base import ToolSpec
from voice_agent.agent.tools.db import READABLE_TABLES
from voice_agent.providers.base import LLM, Message

log = logging.getLogger(__name__)

MAX_ROWS = 20
QUERY_TIMEOUT_S = 5.0

#: Generation budget for the SQL itself. Nothing to do with how much the
#: agent says out loud.
SQL_MAX_TOKENS = 400

SCHEMA_FOR_PROMPT = """
-- CRM
reps(id, name, region)
accounts(id, name, region, segment, rep_id, credit_limit, onboarded_at)
  segment is one of: independent, regional, national
contacts(id, account_id, name, role)
activities(id, account_id, rep_id, kind, occurred_at, summary)
  kind is one of: visit, call, email
notes(id, account_id, written_at, author, body)      -- free text

-- ERP
products(id, sku, name, category, unit_price)
  category is one of: Fixings, Electrical, Plumbing, Timber
orders(id, account_id, ordered_at, status, channel, total)
  status is one of: delivered, shipped, pending
  channel is one of: rep, portal, phone
order_items(id, order_id, product_id, quantity, unit_price)

-- Behavioural (web portal)
portal_sessions(id, account_id, occurred_at, category, duration_s)
quote_requests(id, account_id, product_id, quantity, requested_at, converted)

-- Actions logged by this assistant
actions(id, account_id, kind, due_at, reason, created_at)

Dates are ISO-8601 strings.
""".strip()

_STATEMENT_SPLIT = re.compile(r";\s*\S")
_IDENTIFIER = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.I)
_FENCE = re.compile(r"```(?:sql)?(.*?)```", re.S)
#: Models label their output even when told not to. Stripping the label is
#: cheaper and more reliable than prompting harder against it.
_LABEL = re.compile(r"^\s*(?:sql|query|answer)\s*:\s*", re.I)


class UnsafeQuery(Exception):
    """The generated SQL was rejected before it reached the database."""


def validate(sql: str) -> str:
    """Return the statement if it is safe to run, else raise."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQuery("empty query")
    if _STATEMENT_SPLIT.search(cleaned):
        raise UnsafeQuery("only a single statement is allowed")
    if not re.match(r"^(select|with)\b", cleaned, re.I):
        raise UnsafeQuery("only SELECT queries are allowed")

    referenced = {match.lower() for match in _IDENTIFIER.findall(cleaned)}
    # Common table expressions name themselves in FROM; allow anything defined
    # by a WITH clause in the same statement.
    defined = {
        name.lower() for name in re.findall(r"\bwith\s+([a-zA-Z_][a-zA-Z0-9_]*)", cleaned, re.I)
    }
    unknown = referenced - READABLE_TABLES - defined
    if unknown:
        raise UnsafeQuery(f"unknown or unreadable table(s): {', '.join(sorted(unknown))}")
    return cleaned


def _extract_sql(raw: str) -> str:
    """Pull the statement out of whatever wrapping the model put around it."""
    fenced = _FENCE.search(raw)
    candidate = (fenced.group(1) if fenced else raw).strip()
    return _LABEL.sub("", candidate).strip()


def build(db_path: Path, llm: LLM) -> list[ToolSpec]:
    from voice_agent.agent.tools import db as database

    async def query_business_data(question: str) -> str:
        prompt = (
            "Translate the question into one SQLite SELECT statement.\n"
            "Return only the SQL, with no explanation and no code fence.\n"
            f"Use at most {MAX_ROWS} rows.\n\n"
            f"Schema:\n{SCHEMA_FOR_PROMPT}\n\n"
            f"Question: {question}\nSQL:"
        )
        parts: list[str] = []
        # A budget sized for two spoken sentences truncates a join halfway
        # through a table name; SQL needs room to finish.
        async for delta in llm.stream(
            [Message(role="user", content=prompt)], max_tokens=SQL_MAX_TOKENS
        ):
            text = getattr(delta, "text", None)
            if text:
                parts.append(text)
        sql = _extract_sql("".join(parts))

        try:
            statement = validate(sql)
        except UnsafeQuery as exc:
            log.warning("rejected generated SQL (%s): %r", exc, sql[:400])
            return f"I could not run that safely: {exc}."

        def run() -> str:
            # Read-only at the engine, so even a statement that slipped past
            # validation cannot modify anything.
            with database.connect(db_path, read_only=True) as connection:
                connection.execute(f"PRAGMA busy_timeout = {int(QUERY_TIMEOUT_S * 1000)}")
                rows = connection.execute(statement).fetchmany(MAX_ROWS)
            if not rows:
                return "That returned no results."
            if len(rows) == 1 and len(rows[0]) == 1:
                # The common case for a spoken answer: one number.
                value = rows[0][0]
                return f"The answer is {value}."
            summary = "; ".join(
                ", ".join(f"{key} {row[key]}" for key in row.keys()) for row in rows[:5]
            )
            more = "" if len(rows) <= 5 else f" and {len(rows) - 5} more"
            return f"{summary}{more}."

        try:
            return await asyncio.wait_for(asyncio.to_thread(run), timeout=QUERY_TIMEOUT_S + 1)
        except TimeoutError:
            return "That query took too long, so I stopped it."
        except sqlite3.Error as exc:
            log.warning("query failed: %s (%s)", exc, statement[:200])
            return f"That query failed: {exc}."

    return [
        ToolSpec(
            name="query_business_data",
            description=(
                "Answer a question about customers, products, orders or revenue "
                "by querying the business database. Use this for aggregate or "
                "comparative questions such as totals, counts, best sellers or "
                "spend by region. For one customer's own orders, prefer "
                "get_order_history."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question, in plain English.",
                    }
                },
                "required": ["question"],
            },
            handler=query_business_data,
        )
    ]
