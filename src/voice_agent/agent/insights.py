"""Derived metrics — the part that is actually the product.

The tables underneath are ordinary. What a distributor's systems cannot tell a
rep is whether an account is *behaving differently from itself*: ordering less
often than it used to, buying less than its peers of the same size, or quietly
browsing a category it has never purchased. None of that is stored anywhere. It
is all computed here, against the frozen snapshot date.

Two principles run through the whole module.

**Every score explains itself.** A churn score of 68 is useless to a rep in a
car; "they have gone from ordering every 19 days to 73 days quiet, spend is down
71 percent, and nobody has been in since May" is the thing they can act on. The
number is a sort key, the reasons are the output.

**Peer comparison never names a peer.** Gaps are computed against a cohort and
returned as aggregates, because telling one customer what another customer buys
is a commercial disclosure a real deployment would be sued over. This is a
constraint in the code, not a note in a prompt.
"""

from __future__ import annotations

import itertools
import logging
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime

from voice_agent.agent import locale
from voice_agent.agent.tools.base import ago
from voice_agent.agent.tools.db import as_of

log = logging.getLogger(__name__)

#: Comparison window. A quarter is long enough to absorb a quiet fortnight and
#: short enough that a real change shows up while it still matters.
WINDOW_DAYS = 90

#: A cohort smaller than this is not a peer group, it is an anecdote. Below it,
#: gap analysis declines to answer rather than inventing a benchmark.
MIN_COHORT = 4

#: Fraction of peers that must buy a category before its absence counts as a
#: gap rather than a preference.
GAP_PEER_THRESHOLD = 0.6


@dataclass(frozen=True)
class Cadence:
    """How often this account orders, and whether it is late by its own standard."""

    mean_days: float
    last_order_days_ago: int
    orders_counted: int

    @property
    def is_overdue(self) -> bool:
        # 1.5x its own rhythm, not a fixed threshold: an account that orders
        # weekly and one that orders quarterly are both "quiet" at very
        # different absolute numbers.
        return self.mean_days > 0 and self.last_order_days_ago > self.mean_days * 1.5

    @property
    def overdue_ratio(self) -> float:
        if self.mean_days <= 0:
            return 0.0
        return self.last_order_days_ago / self.mean_days


@dataclass(frozen=True)
class Trend:
    """Trailing window against the one before it."""

    recent: float
    prior: float

    @property
    def change_pct(self) -> float:
        if self.prior <= 0:
            return 0.0
        return (self.recent - self.prior) / self.prior * 100.0

    @property
    def direction(self) -> str:
        if self.change_pct > 12:
            return "up"
        if self.change_pct < -12:
            return "down"
        return "flat"


@dataclass(frozen=True)
class Gap:
    """A category this account's peers buy and it does not."""

    category: str
    peer_share: float
    estimated_annual_value: float


@dataclass(frozen=True)
class Risk:
    """A churn score that carries its own justification."""

    score: int
    reasons: list[str] = field(default_factory=list)

    @property
    def band(self) -> str:
        if self.score >= 60:
            return "high"
        if self.score >= 30:
            return "medium"
        return "low"


@dataclass(frozen=True)
class Brief:
    """Everything needed to walk into a meeting."""

    account_id: int
    name: str
    segment: str
    region: str
    rep: str
    contact: str
    cadence: Cadence
    trend: Trend
    risk: Risk
    gaps: list[Gap]
    intent: list[str]
    last_contact_days: int | None
    last_contact_on: str | None
    last_order_on: str | None
    trailing_year_revenue: float


# ─────────────────────────────────────────────────────────── building blocks


def _days_between(later: datetime, earlier: str) -> int:
    return max(0, (later - datetime.fromisoformat(earlier)).days)


def reorder_cadence(db: sqlite3.Connection, account_id: int) -> Cadence:
    """Mean gap between orders, and how long since the last one."""
    snapshot = as_of(db)
    rows = db.execute(
        "SELECT ordered_at FROM orders WHERE account_id = ? ORDER BY ordered_at DESC LIMIT 25",
        (account_id,),
    ).fetchall()
    if not rows:
        return Cadence(mean_days=0.0, last_order_days_ago=9999, orders_counted=0)

    dates = [datetime.fromisoformat(row["ordered_at"]) for row in rows]
    gaps = [(a - b).days for a, b in itertools.pairwise(dates)]
    # Median, not mean: one 90-day gap over Christmas should not redefine an
    # account's rhythm.
    mean = float(statistics.median(gaps)) if gaps else 0.0
    return Cadence(
        mean_days=mean,
        last_order_days_ago=(snapshot - dates[0]).days,
        orders_counted=len(dates),
    )


def revenue_trend(db: sqlite3.Connection, account_id: int, window: int = WINDOW_DAYS) -> Trend:
    """Spend in the trailing window against the window before it."""
    snapshot = as_of(db).isoformat()
    row = db.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN julianday(:asof) - julianday(ordered_at)
                            <= :w THEN total END), 0) AS recent,
          COALESCE(SUM(CASE WHEN julianday(:asof) - julianday(ordered_at)
                            > :w AND julianday(:asof) - julianday(ordered_at)
                            <= :w2 THEN total END), 0) AS prior
        FROM orders WHERE account_id = :id
        """,
        {"asof": snapshot, "w": window, "w2": window * 2, "id": account_id},
    ).fetchone()
    return Trend(recent=float(row["recent"]), prior=float(row["prior"]))


def last_contact(db: sqlite3.Connection, account_id: int) -> tuple[int, str] | None:
    """Days since anyone spoke to them, and the date it happened.

    Both, always. A caller given only the age invents a date; a caller given
    only a date has to do arithmetic against a snapshot it cannot see.
    """
    row = db.execute(
        "SELECT MAX(occurred_at) AS last FROM activities WHERE account_id = ?",
        (account_id,),
    ).fetchone()
    if row is None or row["last"] is None:
        return None
    return _days_between(as_of(db), row["last"]), row["last"]


def contact_recency(db: sqlite3.Connection, account_id: int) -> int | None:
    """Days since anyone from the business spoke to them."""
    result = last_contact(db, account_id)
    return result[0] if result else None


def trailing_revenue(db: sqlite3.Connection, account_id: int, days: int = 365) -> float:
    snapshot = as_of(db).isoformat()
    row = db.execute(
        "SELECT COALESCE(SUM(total), 0) AS total FROM orders"
        " WHERE account_id = :id AND julianday(:asof) - julianday(ordered_at) <= :d",
        {"id": account_id, "asof": snapshot, "d": days},
    ).fetchone()
    return float(row["total"])


@dataclass(frozen=True, slots=True)
class CategorySpend:
    """What an account actually buys in one category."""

    category: str
    spend: float
    orders: int


def purchase_profile(
    db: sqlite3.Connection, account_id: int, days: int = 365
) -> list[CategorySpend]:
    """What they buy, biggest first.

    The counterpart to :func:`category_gaps`, which only ever answers what an
    account does *not* buy. A rep walking into a meeting asks both questions,
    and until this existed the assistant could only answer one of them.
    """
    snapshot = as_of(db).isoformat()
    rows = db.execute(
        "SELECT p.category AS category,"
        "       SUM(oi.quantity * oi.unit_price) AS spend,"
        "       COUNT(DISTINCT o.id) AS orders"
        " FROM orders o"
        " JOIN order_items oi ON oi.order_id = o.id"
        " JOIN products p ON p.id = oi.product_id"
        " WHERE o.account_id = :id"
        "   AND julianday(:asof) - julianday(o.ordered_at) <= :d"
        " GROUP BY p.category ORDER BY spend DESC",
        {"id": account_id, "asof": snapshot, "d": days},
    ).fetchall()
    return [
        CategorySpend(category=r["category"], spend=float(r["spend"]), orders=int(r["orders"]))
        for r in rows
    ]


def similar_names(db: sqlite3.Connection, name: str, rep: str, limit: int = 3) -> list[str]:
    """The rep's accounts whose names are closest to what was heard.

    A failed lookup that says only "I cannot find that" sends the agent round
    the same search again, which is what a rep experiences as the assistant
    being stuck. Offering the near misses turns a dead end into a choice, and
    it also absorbs the real cause most of the time - a transcription that got
    the company name slightly wrong.

    Scoped to the rep, for the same reason :func:`find_account` is: a helpful
    suggestion naming another rep's customer is still a disclosure.
    """
    import difflib

    mine = [
        str(row["name"])
        for row in db.execute(
            "SELECT a.name FROM accounts a JOIN reps r ON r.id = a.rep_id WHERE r.name = ?",
            (rep,),
        )
    ]
    spoken = name.strip().lower()
    # Cheap substring hits first - "Severn" for "Severn Valley Industrial
    # Fasteners" is a better suggestion than anything edit distance finds.
    hits = [n for n in mine if spoken and spoken in n.lower()]
    if len(hits) < limit:
        close = difflib.get_close_matches(name, mine, n=limit, cutoff=0.4)
        hits += [n for n in close if n not in hits]
    return hits[:limit]


def category_gaps(db: sqlite3.Connection, account_id: int) -> list[Gap]:
    """Categories this account's peer cohort buys and it does not.

    The cohort is same-segment accounts, which controls for size — a national
    chain and a one-branch independent are not peers and comparing them
    produces nonsense recommendations.

    Returns aggregates only. Naming which peers buy what would be disclosing
    one customer's purchasing to another.
    """
    snapshot = as_of(db).isoformat()
    account = db.execute("SELECT segment FROM accounts WHERE id = ?", (account_id,)).fetchone()
    if account is None:
        return []

    peers = [
        row["id"]
        for row in db.execute(
            "SELECT id FROM accounts WHERE segment = ? AND id != ?",
            (account["segment"], account_id),
        )
    ]
    if len(peers) < MIN_COHORT:
        return []

    def category_spend(ids: list[int]) -> dict[int, dict[str, float]]:
        placeholders = ",".join("?" * len(ids))
        rows = db.execute(
            f"""
            SELECT o.account_id AS aid, p.category AS category,
                   SUM(oi.quantity * oi.unit_price) AS spend
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            JOIN products p ON p.id = oi.product_id
            WHERE o.account_id IN ({placeholders})
              AND julianday(?) - julianday(o.ordered_at) <= 365
            GROUP BY o.account_id, p.category
            """,
            (*ids, snapshot),
        ).fetchall()
        out: dict[int, dict[str, float]] = {}
        for row in rows:
            out.setdefault(row["aid"], {})[row["category"]] = float(row["spend"])
        return out

    peer_spend = category_spend(peers)
    mine = category_spend([account_id]).get(account_id, {})

    my_total = sum(mine.values()) or 1.0
    peer_totals = [sum(v.values()) for v in peer_spend.values() if v]
    peer_median_total = statistics.median(peer_totals) if peer_totals else 1.0
    # Scale the benchmark to this account's size, so the estimate is what *they*
    # would plausibly spend rather than what an average peer spends.
    size_ratio = my_total / peer_median_total if peer_median_total else 1.0

    gaps: list[Gap] = []
    categories = {row["category"] for row in db.execute("SELECT DISTINCT category FROM products")}
    for category in sorted(categories):
        if mine.get(category, 0.0) > 0:
            continue
        buyers = [v[category] for v in peer_spend.values() if v.get(category, 0) > 0]
        share = len(buyers) / len(peers)
        if share < GAP_PEER_THRESHOLD:
            continue
        gaps.append(
            Gap(
                category=category,
                peer_share=share,
                estimated_annual_value=round(statistics.median(buyers) * size_ratio, -1),
            )
        )
    return sorted(gaps, key=lambda g: g.estimated_annual_value, reverse=True)


def intent_signals(db: sqlite3.Connection, account_id: int, days: int = 21) -> list[str]:
    """Recent browsing and quoting that has not turned into an order.

    This is the cross-source signal: interest recorded by the portal that the
    ERP has no matching order for. Neither system knows it alone.
    """
    snapshot = as_of(db).isoformat()
    signals: list[str] = []

    sessions = db.execute(
        "SELECT category, COUNT(*) AS n FROM portal_sessions"
        " WHERE account_id = :id AND julianday(:asof) - julianday(occurred_at) <= :d"
        " GROUP BY category ORDER BY n DESC",
        {"id": account_id, "asof": snapshot, "d": days},
    ).fetchall()
    for row in sessions:
        if row["n"] >= 3:
            signals.append(f"{row['n']} portal visits to {row['category']} in the last {days} days")

    quotes = db.execute(
        "SELECT p.name AS name, q.quantity AS quantity FROM quote_requests q"
        " JOIN products p ON p.id = q.product_id"
        " WHERE q.account_id = :id AND q.converted = 0"
        "   AND julianday(:asof) - julianday(q.requested_at) <= :d",
        {"id": account_id, "asof": snapshot, "d": days * 2},
    ).fetchall()
    for row in quotes:
        signals.append(f"unconverted quote for {row['quantity']} x {row['name']}")

    return signals


def churn_risk(db: sqlite3.Connection, account_id: int) -> Risk:
    """A transparent, additive score. Every point has a stated cause."""
    cadence = reorder_cadence(db, account_id)
    trend = revenue_trend(db, account_id)
    contact = last_contact(db, account_id)

    score = 0
    reasons: list[str] = []

    if cadence.is_overdue:
        # Capped so one enormous gap cannot dominate the whole score.
        points = min(40, int((cadence.overdue_ratio - 1.5) * 30) + 20)
        score += points
        reasons.append(
            locale.t(
                "reason_overdue",
                mean=f"{cadence.mean_days:.0f}",
                actual=cadence.last_order_days_ago,
            )
        )

    if trend.direction == "down":
        points = min(35, int(abs(trend.change_pct) / 3))
        score += points
        reasons.append(locale.t("reason_trend", pct=f"{abs(trend.change_pct):.0f}"))

    if contact is not None and contact[0] > 45:
        since_contact, when = contact
        points = min(25, int((since_contact - 45) / 2) + 8)
        score += points
        # The date as well as the age, so the model never has to guess one
        # from the other and state it as fact.
        reasons.append(locale.t("reason_quiet", when=ago(since_contact, when)))

    return Risk(score=min(100, score), reasons=reasons)


def brief(db: sqlite3.Connection, account_id: int) -> Brief | None:
    """Compose everything a rep needs before walking in."""
    row = db.execute(
        "SELECT a.id, a.name, a.segment, a.region, r.name AS rep,"
        "       (SELECT c.name || ' (' || c.role || ')' FROM contacts c"
        "         WHERE c.account_id = a.id LIMIT 1) AS contact"
        " FROM accounts a JOIN reps r ON r.id = a.rep_id WHERE a.id = ?",
        (account_id,),
    ).fetchone()
    if row is None:
        return None

    contact_days, contact_on = last_contact(db, account_id) or (None, None)
    last_order_on = db.execute(
        "SELECT MAX(ordered_at) AS d FROM orders WHERE account_id = ?",
        (account_id,),
    ).fetchone()["d"]

    return Brief(
        account_id=row["id"],
        name=row["name"],
        segment=row["segment"],
        region=row["region"],
        rep=row["rep"],
        contact=row["contact"] or "no named contact",
        cadence=reorder_cadence(db, account_id),
        trend=revenue_trend(db, account_id),
        risk=churn_risk(db, account_id),
        gaps=category_gaps(db, account_id),
        intent=intent_signals(db, account_id),
        last_contact_days=contact_days,
        last_contact_on=contact_on,
        last_order_on=last_order_on,
        trailing_year_revenue=trailing_revenue(db, account_id),
    )


def find_account(db: sqlite3.Connection, name: str, rep: str | None = None) -> int | None:
    """Resolve a spoken account name to an id.

    Transcribed speech rarely matches a registered company name exactly, so
    matching is loose. It is **not** loose about ownership: when a rep is given,
    the search never leaves their book.

    An earlier version fell back to a global search when the rep-scoped match
    missed. That was meant to forgive bad transcription and instead quietly
    forgave the access boundary - the assistant would brief a rep on another
    rep's customer. Loose about spelling, strict about whose account it is.
    """
    pattern = f"%{name.strip()}%"
    if rep:
        row = db.execute(
            "SELECT a.id FROM accounts a JOIN reps r ON r.id = a.rep_id"
            " WHERE a.name LIKE ? AND r.name = ? ORDER BY LENGTH(a.name) LIMIT 1",
            (pattern, rep),
        ).fetchone()
        return int(row["id"]) if row else None

    row = db.execute(
        "SELECT id FROM accounts WHERE name LIKE ? ORDER BY LENGTH(name) LIMIT 1",
        (pattern,),
    ).fetchone()
    return int(row["id"]) if row else None
