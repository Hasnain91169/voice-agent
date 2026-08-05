"""Natural-language querying over the business data.

The interesting tool in the suite, and the dangerous one: the model writes SQL
and the server runs it. Three independent guards, because any one of them alone
is a bad bet:

1. **The connection cannot write.** Opened with SQLite's ``mode=ro`` URI, so a
   DROP or UPDATE fails at the engine regardless of what got past inspection.
   Pattern-matching statements is always one clever payload from being wrong;
   a connection that physically cannot write is not.
2. **Rep-scoped temporary views.** The model sees only views whose rows are
   filtered to the current rep's book. Base tables are not in the generated
   SQL schema, so a query cannot widen scope by omitting a WHERE clause.
3. **One statement, SELECT only, allowlisted views.** Catches the obvious
   cases early and with a comprehensible error the model can recover from.
4. **A row cap and a timeout.** A cross join over the order items would
   otherwise stall a live phone call.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from pathlib import Path

from voice_agent.agent.tools.base import ToolSpec
from voice_agent.providers.base import LLM, Message

log = logging.getLogger(__name__)

MAX_ROWS = 20
QUERY_TIMEOUT_S = 5.0

#: Generation budget for the SQL itself. Nothing to do with how much the
#: agent says out loud.
SQL_MAX_TOKENS = 400

SCOPED_READABLE_TABLES = frozenset(
    {
        "my_rep",
        "my_accounts",
        "my_contacts",
        "my_activities",
        "my_notes",
        "my_orders",
        "my_order_items",
        "my_portal_sessions",
        "my_quote_requests",
        "my_actions",
        "catalog_products",
        "snapshot_meta",
    }
)

SCHEMA_FOR_PROMPT = """
-- CRM views, already limited to the current rep's patch
my_rep(id, name, region)
my_accounts(id, name, region, segment, rep_id, credit_limit, onboarded_at)
  segment is one of: independent, regional, national
my_contacts(id, account_id, name, role)
my_activities(id, account_id, rep_id, kind, occurred_at, summary)
  kind is one of: visit, call, email
my_notes(id, account_id, written_at, author, body)   -- free text

-- ERP views, already limited through my_accounts
catalog_products(id, sku, name, category, unit_price)
  category is one of: Fixings, Electrical, Plumbing, Timber
my_orders(id, account_id, ordered_at, status, channel, total)
  status is one of: delivered, shipped, pending
  channel is one of: rep, portal, phone
my_order_items(id, order_id, product_id, quantity, unit_price)

-- Behavioural (web portal)
my_portal_sessions(id, account_id, occurred_at, category, duration_s)
my_quote_requests(id, account_id, product_id, quantity, requested_at, converted)

-- Actions logged by this assistant
my_actions(id, account_id, kind, due_at, reason, created_at)
snapshot_meta(key, value)

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
    unknown = referenced - SCOPED_READABLE_TABLES - defined
    if unknown:
        raise UnsafeQuery(f"unknown or unreadable table(s): {', '.join(sorted(unknown))}")
    return cleaned


def _extract_sql(raw: str) -> str:
    """Pull the statement out of whatever wrapping the model put around it."""
    fenced = _FENCE.search(raw)
    candidate = (fenced.group(1) if fenced else raw).strip()
    return _LABEL.sub("", candidate).strip()


def _create_scoped_views(connection: sqlite3.Connection, rep: str) -> None:
    """Expose only the current rep's rows to generated SQL.

    The main database stays read-only. These temporary views live in SQLite's
    in-memory temp schema and are created by trusted code before the generated
    statement runs. The model only receives their names, never the underlying
    base-table schema.
    """
    row = connection.execute("SELECT id FROM main.reps WHERE name = ?", (rep,)).fetchone()
    rep_id = int(row["id"]) if row is not None else -1
    definitions = {
        "my_rep": f"SELECT id, name, region FROM main.reps WHERE id = {rep_id}",
        "my_accounts": (
            "SELECT id, name, region, segment, rep_id, credit_limit, onboarded_at "
            f"FROM main.accounts WHERE rep_id = {rep_id}"
        ),
        "my_contacts": (
            "SELECT id, account_id, name, role FROM main.contacts "
            "WHERE account_id IN (SELECT id FROM my_accounts)"
        ),
        "my_activities": (
            "SELECT id, account_id, rep_id, kind, occurred_at, summary "
            f"FROM main.activities WHERE rep_id = {rep_id} "
            "AND account_id IN (SELECT id FROM my_accounts)"
        ),
        "my_notes": (
            "SELECT id, account_id, written_at, author, body FROM main.notes "
            "WHERE account_id IN (SELECT id FROM my_accounts)"
        ),
        "my_orders": (
            "SELECT id, account_id, ordered_at, status, channel, total "
            "FROM main.orders WHERE account_id IN (SELECT id FROM my_accounts)"
        ),
        "my_order_items": (
            "SELECT oi.id, oi.order_id, oi.product_id, oi.quantity, oi.unit_price "
            "FROM main.order_items oi JOIN main.orders o ON o.id = oi.order_id "
            "WHERE o.account_id IN (SELECT id FROM my_accounts)"
        ),
        "my_portal_sessions": (
            "SELECT id, account_id, occurred_at, category, duration_s "
            "FROM main.portal_sessions WHERE account_id IN (SELECT id FROM my_accounts)"
        ),
        "my_quote_requests": (
            "SELECT id, account_id, product_id, quantity, requested_at, converted "
            "FROM main.quote_requests WHERE account_id IN (SELECT id FROM my_accounts)"
        ),
        "my_actions": (
            "SELECT id, account_id, kind, due_at, reason, created_at "
            "FROM main.actions WHERE account_id IN (SELECT id FROM my_accounts)"
        ),
        "catalog_products": (
            "SELECT id, sku, name, category, unit_price FROM main.products"
        ),
        "snapshot_meta": "SELECT key, value FROM main.meta",
    }
    for name, statement in definitions.items():
        connection.execute(f"CREATE TEMP VIEW {name} AS {statement}")


def build(db_path: Path, llm: LLM, *, rep: str) -> list[ToolSpec]:
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
                _create_scoped_views(connection, rep)
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
