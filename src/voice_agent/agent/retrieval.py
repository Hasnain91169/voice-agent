"""Retrieval over the free-text visit notes.

Everything else the assistant knows is computed from tables. This module is the
part that answers questions the schema cannot: *"has anyone complained about
deliveries?"*, *"which of my accounts have mentioned a competitor?"*, *"what did
we say we would do for them?"* — where the answer only exists in prose a rep
typed into their phone.

Three decisions shape it.

**BM25 over SQLite FTS5, not a vector database.** The corpus is a few hundred
short notes per rep. At that size an embedding index is slower to build, another
dependency to ship, and another thing to keep consistent with the source of
truth — for a set small enough to fit in a CPU cache. It reaches 79% recall@3
against deliberately paraphrased questions at a fifth of a millisecond, measured
in ``bench/retrieval.py``. What BM25 cannot do is negation: "who is unhappy"
retrieves "not one complaint today", because the words match and the polarity is
invisible to it. Quoting the note verbatim is what contains that — the model
reads the "not" even though the index could not.

**The query is widened with domain synonyms, after that decision was got wrong
once.** A hand-written table mapping "complaint" to "frustrated" and "competitor"
to "rival" measured as worth nothing on 80 paraphrased probes — net zero, one
gained and one lost — so it was deleted, with a confident note saying domain
intuition is worth what it measures.

The measurement was the problem, not the intuition. Those probes ask the
retriever to *find one specific note*, and a rep also asks it to *sweep* — "has
anyone mentioned a competitor?" — where nine notes are relevant, they use nine
different words, and three slots are available. Nobody had measured that, and
the agent duly searched the single word "competitor", surfaced one note of nine,
and told a rep "not recently, no".

With both tasks measured (``bench/retrieval.py``), expansion costs two points of
lookup recall@3 — 76% to 74%, which is two probes — and gains nineteen points of
sweep hit rate, 54% to 73%, while *improving* sweep precision from 26% to 36%.
Widening finds more of the right notes than wrong ones, so it is on.

**The query arrives from a speech recogniser.** It is lower case, unpunctuated,
and contains words the corpus does not: "erm", "have they said anything about".
So the query is stripped to content terms and OR-ed rather than AND-ed — an
utterance that must match every token returns nothing, which reads to the caller
as "there is nothing there" when the truth is "you said an extra word".

**Results are quoted, never summarised.** A retrieved note is returned verbatim
with its author and date attached. The whole value of a note is that a person
wrote it; a paraphrase of a paraphrase is how "frustrated with lead times last
month" becomes "they are threatening to leave".
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass

from voice_agent.agent.tools.base import spoken_date

log = logging.getLogger(__name__)

#: Words that carry no retrieval signal but appear constantly in spoken queries.
#: Deliberately short: an aggressive stop list removes terms that matter in this
#: domain ("out" of stock, "on" account).
_STOP = frozenset(
    """a an and any anything are as at about be been being but by can did do does
    for from get got had has have he her him his how i if in into is it its me my
    of on or our out said say says she so some tell that the their them then there
    these they this to us was we were what when where which who will with would
    you your me know anyone anybody ever""".split()
)

#: Terms a rep says mapped to terms the corpus uses. On by default; the flag
#: survives so ``bench/retrieval.py`` can keep both answers visible, because
#: "does query expansion help?" has a different answer for lookups than for
#: sweeps and only one of those was measured the first time.
#:
#: Stems, not words: FTS5 applies porter stemming to the corpus, so "frustrat"
#: covers frustrated and frustrating without a second entry.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "competitor": ("rival", "elsewhere", "switch", "trialling", "undercut", "another"),
    "competitors": ("rival", "elsewhere", "switch", "undercut"),
    "elsewhere": ("rival", "competitor", "switch", "another"),
    "complaint": ("frustrat", "unhappy", "annoyed", "moaning", "grumbl", "issue"),
    "complaining": ("frustrat", "unhappy", "annoyed", "moaning", "grumbl"),
    "unhappy": ("frustrat", "annoyed", "moaning", "grumbl", "complaint"),
    "moving": ("switch", "elsewhere", "rival", "leaving"),
    "delivery": ("deliver", "lead", "late", "delay"),
    "deliveries": ("deliver", "lead", "late", "delay"),
    "price": ("pricing", "quote", "discount", "undercut", "cheaper", "terms"),
    "credit": ("terms", "invoice", "settlement", "payment", "finance"),
    "payment": ("invoice", "terms", "settlement", "paying"),
    "staff": ("manager", "buyer", "starting", "leaving", "director"),
    "buyer": ("manager", "contact", "director", "estimator"),
    "job": ("site", "contract", "development", "tender", "scheme"),
    "tender": ("job", "contract", "site", "quote"),
    "expanding": ("expansion", "branch", "depot", "growing", "opening"),
    "quiet": ("slowed", "dropped", "down", "nothing", "stopped"),
    "promised": ("promis", "chase", "owe", "said", "follow"),
    "follow": ("chase", "promis", "diaris", "action"),
}

#: Above this many results a spoken answer stops being an answer.
MAX_RESULTS = 3


@dataclass(frozen=True, slots=True)
class Note:
    """One retrieved note, with everything needed to attribute it."""

    account_id: int
    account: str
    author: str
    written_at: str
    body: str
    score: float

    def spoken(self, *, with_account: bool) -> str:
        """The note as it should be read out.

        The date and author come first because they are what make it evidence
        rather than assertion — "Dani wrote in May" is checkable, "they were
        unhappy" is not.
        """
        who = f"{self.account}: " if with_account else ""
        return f"{who}{self.author} wrote on {spoken_date(self.written_at)}: {self.body}"


def terms(query: str, *, expand: bool = True) -> list[str]:
    """Content words from a spoken query, optionally widened with synonyms."""
    words = [w for w in re.findall(r"[a-z0-9']+", query.lower()) if w not in _STOP and len(w) > 2]
    if not expand:
        return list(dict.fromkeys(words))
    widened: list[str] = []
    for word in words:
        widened.append(word)
        widened.extend(_SYNONYMS.get(word, ()))
    return list(dict.fromkeys(widened))


def _match_expression(
    query: str, *, conjunction: str = "OR", prefix: bool = False, expand: bool = True
) -> str:
    """An FTS5 MATCH expression that is forgiving about extra words.

    OR, not AND, and this is not a tuning preference. Requiring every term
    returns **nothing at all** — 0 of 80 probes, not one — because a spoken
    question always carries a word the corpus does not have, and one such word
    is enough to empty an AND. The caller would hear "there is no record of
    that" when the record exists.

    Prefix matching defaults off. It measured identically at recall@3 and @10
    and one probe worse at @1, so on the same standard that removed the synonym
    table it does not earn its place. Porter stemming already covers the
    morphology it was meant to catch.

    Both knobs stay as parameters so ``bench/retrieval.py`` can hold these
    claims to account rather than restate them.
    """
    found = terms(query, expand=expand)
    if not found:
        return ""
    star = "*" if prefix else ""
    # Quoted so that terms containing an apostrophe cannot be read as syntax.
    return f" {conjunction} ".join(f'"{term}"{star}' for term in found)


def search(
    db: sqlite3.Connection,
    query: str,
    rep: str,
    *,
    account_id: int | None = None,
    limit: int = MAX_RESULTS,
    conjunction: str = "OR",
    prefix: bool = False,
    expand: bool = True,
) -> list[Note]:
    """The rep's notes most relevant to a question, best first.

    Always scoped to the rep. The notes contain named individuals, site
    addresses and commercial grievances, so a retrieval tool that reaches across
    the whole company is a data-protection incident with a natural-language
    interface. The join to ``reps`` is what enforces it, not the prompt.
    """
    expression = _match_expression(query, conjunction=conjunction, prefix=prefix, expand=expand)
    if not expression:
        return []

    sql = (
        "SELECT n.account_id, a.name AS account, n.author, n.written_at, n.body,"
        "       bm25(notes_fts) AS score"
        " FROM notes_fts"
        " JOIN notes n ON n.id = notes_fts.rowid"
        " JOIN accounts a ON a.id = n.account_id"
        " JOIN reps r ON r.id = a.rep_id"
        " WHERE notes_fts MATCH ? AND r.name = ?"
    )
    params: list[object] = [expression, rep]
    if account_id is not None:
        sql += " AND n.account_id = ?"
        params.append(account_id)
    # bm25() returns a negative score, better matches being more negative.
    sql += " ORDER BY score LIMIT ?"
    params.append(limit)

    try:
        rows = db.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        # A malformed MATCH expression is a bug here, not a caller error, but it
        # must not take the turn down with it.
        log.warning("note search failed for %r: %s", query, exc)
        return []

    return [
        Note(
            account_id=int(row["account_id"]),
            account=str(row["account"]),
            author=str(row["author"]),
            written_at=str(row["written_at"]),
            body=str(row["body"]),
            score=float(row["score"]),
        )
        for row in rows
    ]
