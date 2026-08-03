"""Tests for note retrieval.

Quality is measured in ``bench/retrieval.py`` against a probe set, not asserted
here — recall is a number that moves, and pinning it to an exact value in a unit
test turns every legitimate improvement into a failing build.

What is pinned here is behaviour that must never change regardless of recall:
the search cannot leave the rep's patch, an empty or stop-word-only query
returns nothing rather than everything, and a retrieved note comes back
byte-for-byte. The notes name individuals, describe commercial grievances and
give site addresses, so a retriever that reaches across the whole company is a
data-protection incident with a natural-language interface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_agent.agent import retrieval
from voice_agent.agent.tools import db as database

REP = "Dani Brooks"
OTHER_REP = "Samir Haq"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return database.seed(tmp_path_factory.mktemp("data") / "wholesale.db")


# ────────────────────────────────────────────────────────── query preparation


def test_terms_drops_spoken_filler() -> None:
    found = retrieval.terms("what have they said about the deliveries")
    assert "deliveries" in found
    assert not {"what", "have", "they", "said", "about", "the"} & set(found)


def test_terms_drops_very_short_words() -> None:
    """Two-letter tokens match everything and rank nothing."""
    assert retrieval.terms("is it on") == []


def test_a_query_of_only_filler_finds_nothing(db_path: Path) -> None:
    """Better silence than the whole corpus.

    An empty MATCH expression would otherwise become "return everything", and
    the agent would read out three unrelated notes as though they answered the
    question.
    """
    with database.connect(db_path, read_only=True) as db:
        assert retrieval.search(db, "what about it", REP) == []
        assert retrieval.search(db, "", REP) == []


# ──────────────────────────────────────────────────────────────── the boundary


def test_search_never_leaves_the_reps_patch(db_path: Path) -> None:
    with database.connect(db_path, read_only=True) as db:
        mine = {
            str(row["name"])
            for row in db.execute(
                "SELECT a.name FROM accounts a JOIN reps r ON r.id = a.rep_id WHERE r.name = ?",
                (REP,),
            )
        }
        # Broad queries are the dangerous ones: a narrow query that happens to
        # match nothing elsewhere proves nothing about scoping.
        for query in ("delivery site order price manager", "account", "new job starting"):
            for note in retrieval.search(db, query, REP, limit=25):
                assert note.account in mine, f"{query!r} leaked {note.account}"


def test_the_same_query_returns_different_notes_for_different_reps(db_path: Path) -> None:
    """Proof the scoping filter is doing work rather than the query being narrow."""
    with database.connect(db_path, read_only=True) as db:
        mine = {n.body for n in retrieval.search(db, "delivery order site", REP, limit=10)}
        theirs = {n.body for n in retrieval.search(db, "delivery order site", OTHER_REP, limit=10)}
    assert mine and theirs
    assert not mine & theirs


# ─────────────────────────────────────────────────────────────── faithfulness


def test_results_are_verbatim(db_path: Path) -> None:
    """The value of a note is that a person wrote it.

    Nothing between the database and the caller may reword it — negation in
    particular survives only because the text does. BM25 happily returns "not
    one complaint today" for "who is unhappy", and quoting it is what lets the
    model see the "not".
    """
    with database.connect(db_path, read_only=True) as db:
        bodies = {
            str(row["body"])
            for row in db.execute(
                "SELECT n.body FROM notes n JOIN accounts a ON a.id = n.account_id"
                " JOIN reps r ON r.id = a.rep_id WHERE r.name = ?",
                (REP,),
            )
        }
        for note in retrieval.search(db, "delivery price manager site", REP, limit=10):
            assert note.body in bodies


def test_spoken_form_attributes_the_note(db_path: Path) -> None:
    with database.connect(db_path, read_only=True) as db:
        hits = retrieval.search(db, "delivery", REP, limit=1)
    assert hits
    spoken = hits[0].spoken(with_account=True)
    assert hits[0].author in spoken
    assert hits[0].account in spoken
    assert hits[0].body in spoken
    assert " of " in spoken, "the date should be spoken, not an ISO string"


# ──────────────────────────────────────────────────────── the corpus itself


def test_no_note_is_shared_between_accounts_on_a_patch(db_path: Path) -> None:
    """Notes carry specific invented people.

    Reusing one across two companies is visibly wrong in the data and worse in
    retrieval: the same text comes back under two accounts and reads as two
    independent pieces of evidence.

    Asserted *within a rep's patch* rather than globally, because that is the
    only view anything ever has — every query joins through the rep. The
    planted injection payloads are deliberately identical across patches, which
    is how one piece of hostile text actually arrives at several customers, and
    no rep can see more than their own copy.
    """
    with database.connect(db_path, read_only=True) as db:
        shared = db.execute(
            "SELECT COUNT(*) FROM ("
            " SELECT n.body, a.rep_id FROM notes n JOIN accounts a ON a.id = n.account_id"
            " GROUP BY n.body, a.rep_id HAVING COUNT(DISTINCT n.account_id) > 1)"
        ).fetchone()[0]
    assert shared == 0


def test_the_corpus_is_actually_varied(db_path: Path) -> None:
    """The first version of this seeder produced 267 notes containing nine
    distinct strings, which no retriever can be evaluated against."""
    with database.connect(db_path, read_only=True) as db:
        row = db.execute(
            "SELECT COUNT(*) c, COUNT(DISTINCT n.body) d FROM notes n"
            " JOIN accounts a ON a.id = n.account_id"
            " JOIN reps r ON r.id = a.rep_id WHERE r.name = ?",
            (REP,),
        ).fetchone()
    assert row["d"] == row["c"], "every note on a patch should be distinct"
    assert row["c"] > 40


def test_no_unfilled_placeholders_reach_the_corpus(db_path: Path) -> None:
    """A note read aloud as "brace contact brace" is worse than no note."""
    with database.connect(db_path, read_only=True) as db:
        leftover = db.execute("SELECT COUNT(*) FROM notes WHERE body LIKE '%{%'").fetchone()[0]
    assert leftover == 0


def test_a_lone_result_says_so(db_path: Path) -> None:
    """One keyword hit is not an answer to "has anyone mentioned X".

    The agent searched the single word "competitor", got one note that happened
    to say there was *no* competitor product on site, and reported "not
    recently, no". The retriever cannot know what it failed to match, but it
    does know how little it found.

    The query is derived from the corpus rather than written here: a word that
    matches exactly one note today may match several after the synonym table
    changes, and a test that silently stops exercising its own subject is worse
    than no test.
    """
    import asyncio
    from collections import Counter

    from voice_agent.agent.tools import sales

    with database.connect(db_path, read_only=True) as db:
        bodies = [
            str(row["body"])
            for row in db.execute(
                "SELECT n.body FROM notes n JOIN accounts a ON a.id = n.account_id"
                " JOIN reps r ON r.id = a.rep_id WHERE r.name = ?",
                (REP,),
            )
        ]
        counts: Counter[str] = Counter()
        for body in bodies:
            counts.update(set(retrieval.terms(body, expand=False)))
        rare = next(
            word
            for word, count in counts.most_common()[::-1]
            if count == 1 and len(retrieval.search(db, word, REP, limit=5)) == 1
        )

    tools = {spec.name: spec for spec in sales.build(db_path, REP)}
    said = asyncio.run(tools["search_notes"].handler(question=rare))
    assert "trying different wording" in said, said


def test_expansion_widens_a_sweep(db_path: Path) -> None:
    """The property the synonym table exists for.

    Measured properly in ``bench/retrieval.py``; asserted here only so that
    turning expansion off by accident fails the build rather than quietly
    costing nineteen points of sweep hit rate.
    """
    with database.connect(db_path, read_only=True) as db:
        narrow = retrieval.search(db, "competitor", REP, limit=10, expand=False)
        wide = retrieval.search(db, "competitor", REP, limit=10)
    assert len(wide) > len(narrow)
