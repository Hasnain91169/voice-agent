"""The note corpus is an untrusted input, and these tests treat it as one.

Visit notes are free text. Reps paste customer emails into them wholesale, and
the agent reads notes aloud after retrieving them — so any instruction sitting
in that text arrives in the model's context wearing the same clothes as
legitimate data. An industrial CRM ingesting supplier correspondence has exactly
this surface, and "the model probably won't fall for it" is not a control.

What is asserted here is the half that does not depend on the model at all. The
tools cannot be talked into leaking a peer, quoting a price, or confirming an
action, because no tool can do those things regardless of what it is asked —
there is no pricing tool to call, gap analysis returns aggregates, and every
lookup joins through the rep. A payload that lands in the model's context still
has nothing to unlock.

Whether the *model* also declines is a separate question, and a probabilistic
one, so it is measured in the eval suite rather than asserted here.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from voice_agent.agent import retrieval
from voice_agent.agent.tools import db as database
from voice_agent.agent.tools import sales

REP = "Dani Brooks"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return database.seed(tmp_path_factory.mktemp("data") / "wholesale.db")


def payloads(db_path: Path, rep: str = REP) -> list[tuple[str, str, str]]:
    with database.connect(db_path, read_only=True) as db:
        return [
            (str(r["attack"]), str(r["name"]), str(r["body"]))
            for r in db.execute(
                "SELECT s.attack, a.name, n.body FROM seed_injections s"
                " JOIN accounts a ON a.id = s.account_id"
                " JOIN notes n ON n.id = s.note_id"
                " JOIN reps rp ON rp.id = a.rep_id"
                " WHERE rp.name = ?",
                (rep,),
            )
        ]


def test_every_attack_is_planted_on_every_patch(db_path: Path) -> None:
    """Partial coverage of a security check reads as coverage until it matters."""
    with database.connect(db_path, read_only=True) as db:
        reps = [str(r["name"]) for r in db.execute("SELECT name FROM reps")]
    attacks = {a for a, _, _ in payloads(db_path)}
    assert attacks
    for rep in reps:
        assert {a for a, _, _ in payloads(db_path, rep)} == attacks, rep


def test_the_answer_key_is_not_readable(db_path: Path) -> None:
    assert "seed_injections" not in database.READABLE_TABLES


def test_payloads_actually_reach_the_model(db_path: Path) -> None:
    """An injection nobody can retrieve tests nothing.

    If the retriever never surfaces these notes, the whole red-team suite
    becomes a very elaborate way of asserting that unreachable text is harmless.
    """
    with database.connect(db_path, read_only=True) as db:
        for _, account, body in payloads(db_path):
            account_id = db.execute(
                "SELECT id FROM accounts WHERE name = ?", (account,)
            ).fetchone()["id"]
            hits = retrieval.search(db, body[:60], REP, account_id=account_id, limit=5)
            assert any(h.body == body for h in hits), account


# ─────────────────────────────────────── the guardrails, under instruction


def test_no_tool_can_quote_a_price(db_path: Path) -> None:
    """The payload grants pricing authority the system does not have to give.

    There is no pricing tool. The attack fails on the tool surface, not on the
    model's judgement, which is the only version of this that holds every time.
    """
    names = {spec.name for spec in sales.build(db_path, REP)}
    assert not {"approve_discount", "quote_price", "set_price"} & names


def test_gap_analysis_still_refuses_to_name_peers(db_path: Path) -> None:
    """The peer_disclosure payload asks for exactly what this tool withholds."""
    tools = {spec.name: spec for spec in sales.build(db_path, REP)}
    with database.connect(db_path, read_only=True) as db:
        everyone = {str(r["name"]) for r in db.execute("SELECT name FROM accounts")}

    for attack, account, _ in payloads(db_path):
        if attack != "peer_disclosure":
            continue
        said = asyncio.run(tools["find_opportunities"].handler(name=account))
        leaked = [n for n in everyone if n != account and n.lower() in said.lower()]
        assert not leaked, leaked


def test_retrieval_stays_rep_scoped_when_the_note_says_otherwise(db_path: Path) -> None:
    """ "They are cleared for it" is text in a row, not a permission grant."""
    with database.connect(db_path, read_only=True) as db:
        mine = {
            str(r["name"])
            for r in db.execute(
                "SELECT a.name FROM accounts a JOIN reps r ON r.id = a.rep_id WHERE r.name = ?",
                (REP,),
            )
        }
        for _, _, body in payloads(db_path):
            # Query using the payload's own words, which is the best case the
            # attacker gets: their text ranks top and still cannot widen scope.
            for note in retrieval.search(db, body, REP, limit=25):
                assert note.account in mine, note.account
