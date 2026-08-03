"""Tests for the sales tools and the insight layer beneath them.

Two kinds of assertion live here, and the second kind is the interesting one.

The ordinary kind checks a tool returns the right facts. The other kind checks
what a tool result **invites the model to say next** — because a result string is
not read by a person, it is read by a language model that will fill any gap in it
with something plausible. "a call 15 days ago" does not state who was on the
call, so the model says "you spoke to them". "77 days" does not state a date, so
the model computes one against its own idea of today, which is its training
cutoff rather than this dataset's frozen snapshot, and states the wrong month as
fact. Both were caught by the eval judge before these tests existed; they are
pinned here so they stay fixed.

The guardrail tests are the same idea one level up. Peer anonymity and rep
scoping are enforced in these functions rather than requested in the prompt, so
they are asserted against the functions.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from voice_agent.agent import insights
from voice_agent.agent.tools import db as database
from voice_agent.agent.tools import sales
from voice_agent.agent.tools.base import ago, money, spoken_date

REP = "Dani Brooks"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A freshly seeded database, built once for the module.

    Seeding is deterministic, so one database serves every test and none of them
    can affect another's reads. The one tool that writes is given its own copy.
    """
    return database.seed(tmp_path_factory.mktemp("data") / "wholesale.db")


@pytest.fixture
def tools(db_path: Path) -> dict[str, object]:
    return {spec.name: spec for spec in sales.build(db_path, REP)}


def account_named(db_path: Path, archetype: str) -> str:
    """One of this rep's accounts generated with the given behaviour."""
    with database.connect(db_path, read_only=True) as db:
        row = db.execute(
            "SELECT a.name FROM accounts a"
            " JOIN reps r ON r.id = a.rep_id"
            " JOIN seed_truth s ON s.account_id = a.id"
            " WHERE r.name = ? AND s.archetype = ? LIMIT 1",
            (REP, archetype),
        ).fetchone()
    assert row is not None, f"the seed planted no {archetype} account for {REP}"
    return str(row["name"])


async def call(tools: dict[str, object], tool: str, **kwargs: object) -> str:
    return str(await tools[tool].handler(**kwargs))  # type: ignore[attr-defined]


# ───────────────────────────────────────────────────── spoken date formatting


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
    ],
)
def test_ordinal_suffixes(day: int, expected: str) -> None:
    assert spoken_date(datetime(2026, 5, day, tzinfo=UTC)) == f"the {expected} of May"


def test_day_is_never_zero_padded() -> None:
    """ "the 03" is read aloud as "the oh three"."""
    assert "0" not in spoken_date(datetime(2026, 5, 3, tzinfo=UTC))


def test_ago_states_the_date_and_the_age() -> None:
    """Both, always — the model can derive neither from the other."""
    said = ago(77, datetime(2026, 5, 17, tzinfo=UTC))
    assert "17th of May" in said
    assert "77 days ago" in said


# ─────────────────────────────────────────── what the result invites next


async def test_activity_names_who_made_contact(tools: dict[str, object], db_path: Path) -> None:
    """The participant is stated, so the model does not have to assume one.

    Every activity in the seed belongs to the account's own rep, so the correct
    rendering is "you" — but that has to come out of a join, not out of the
    model's imagination.
    """
    said = await call(tools, "get_recent_activity", name=account_named(db_path, "at_risk"))
    assert said.startswith("You had a")


async def test_activity_dates_every_relative_age(tools: dict[str, object], db_path: Path) -> None:
    said = await call(tools, "get_recent_activity", name=account_named(db_path, "at_risk"))
    for age in re.finditer(r"(\d+) days ago", said):
        # Each age is preceded by the date it resolves to.
        assert re.search(r"the \d+\w\w of \w+, " + age.group(1) + " days ago", said), said


async def test_brief_never_gives_a_bare_day_count(tools: dict[str, object], db_path: Path) -> None:
    """A day count with no date is what produced the invented "early June"."""
    said = await call(tools, "brief_account", name=account_named(db_path, "at_risk"))
    ages = re.findall(r"(\d+) days ago", said)
    assert ages, "the at-risk brief should mention recency at all"
    for age in ages:
        assert re.search(r"the \d+\w\w of \w+, " + age + " days ago", said), said


async def test_risk_reasons_carry_dates(tools: dict[str, object], db_path: Path) -> None:
    """The reasons are read aloud verbatim, so they need the same treatment."""
    said = await call(tools, "find_accounts_at_risk", limit=3)
    if "nobody has been in touch" in said:
        assert re.search(r"in touch since the \d+\w\w of \w+, \d+ days ago", said), said


async def test_patch_summary_answers_all_accounts(tools: dict[str, object]) -> None:
    """The transcript failure was a missing portfolio-level tool."""
    said = await call(tools, "summarize_patch")

    assert "Across your patch" in said
    assert "accounts" in said
    assert "Risk is" in said or "Nothing is showing material churn risk" in said
    assert "I don't have a tool" not in said


async def test_patch_summary_is_rep_scoped(tools: dict[str, object], db_path: Path) -> None:
    said = await call(tools, "summarize_patch")

    with database.connect(db_path, read_only=True) as db:
        others = {
            str(row["name"])
            for row in db.execute(
                "SELECT a.name FROM accounts a JOIN reps r ON r.id = a.rep_id WHERE r.name != ?",
                (REP,),
            )
        }
    leaked = [name for name in others if name.lower() in said.lower()]
    assert not leaked, f"leaked another rep's accounts: {leaked}"


# ───────────────────────────────────────────────────────────────── guardrails


def test_find_account_does_not_cross_reps(db_path: Path) -> None:
    """Loose about spelling, strict about ownership.

    A fuzzy match that falls back to a global search is a plausible-looking
    convenience that quietly hands one rep another rep's book.
    """
    with database.connect(db_path, read_only=True) as db:
        other = db.execute(
            "SELECT a.name FROM accounts a JOIN reps r ON r.id = a.rep_id"
            " WHERE r.name != ? LIMIT 1",
            (REP,),
        ).fetchone()
        assert insights.find_account(db, str(other["name"]), REP) is None


async def test_opportunities_never_name_a_peer(tools: dict[str, object], db_path: Path) -> None:
    """Gap analysis returns aggregates. Which peers is a commercial disclosure."""
    with database.connect(db_path, read_only=True) as db:
        every_account = [str(row["name"]) for row in db.execute("SELECT name FROM accounts")]

    target = account_named(db_path, "category_gap")
    said = await call(tools, "find_opportunities", name=target)
    named = [name for name in every_account if name != target and name.lower() in said.lower()]
    assert not named, f"leaked peer names: {named}"


async def test_no_tool_can_read_the_answer_key(db_path: Path) -> None:
    """``seed_truth`` is how the evals know the truth. A readable one would let
    the agent mark its own homework."""
    assert "seed_truth" not in database.READABLE_TABLES


# ────────────────────────────────────────────────── spoken figures


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (7160, "7,200 pounds"),
        (450.4, "450 pounds"),
        (34210, "34 thousand pounds"),
        (1_250_000, "1.2 million pounds"),
    ],
)
def test_money_is_already_rounded(amount: float, expected: str) -> None:
    """The tool does the arithmetic so the model does not have to.

    "7,160.00 pounds" is what produced a spoken "seventy-nine hundred": the
    model had to strip the precision itself and mis-rounded doing it.
    """
    assert money(amount) == expected


def test_money_never_returns_decimals() -> None:
    """A synthesiser reads ".00" aloud, and the model rounds it away by hand."""
    assert all("." not in money(v) for v in (7160.55, 999.99, 12345.67))


# ──────────────────────────────────────────── what they buy, and near misses


async def test_purchase_profile_names_categories_they_buy(
    tools: dict[str, object], db_path: Path
) -> None:
    """The question find_opportunities cannot answer."""
    name = account_named(db_path, "healthy")
    said = await call(tools, "get_purchase_profile", name=name)
    with database.connect(db_path, read_only=True) as db:
        account_id = insights.find_account(db, name, REP)
        assert account_id is not None
        profile = insights.purchase_profile(db, account_id)
    assert profile, "a healthy account should have bought something"
    assert profile[0].category.lower() in said.lower()


async def test_failed_lookup_offers_near_misses(tools: dict[str, object], db_path: Path) -> None:
    """A garbled company name is the common case, not an exotic one."""
    real = account_named(db_path, "healthy")
    garbled = real[:4] + "x " + real.split()[-1]
    said = await call(tools, "brief_account", name=garbled)
    assert real in said, said


def test_near_misses_never_cross_reps(db_path: Path) -> None:
    """The suggestion path emits account names, so it needs the same boundary
    as ``find_account`` — a helpful suggestion naming another rep's customer is
    still a disclosure."""
    with database.connect(db_path, read_only=True) as db:
        others = {
            str(row["name"])
            for row in db.execute(
                "SELECT a.name FROM accounts a JOIN reps r ON r.id = a.rep_id WHERE r.name != ?",
                (REP,),
            )
        }
        for probe in ("Industrial", "Timber", "a", "Ltd"):
            assert not set(insights.similar_names(db, probe, REP)) & others
