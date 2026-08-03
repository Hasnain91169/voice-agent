"""Tests for per-turn language, and for the line between chrome and data.

The interesting assertions here are about what *stays English*. Account names,
contact names, activity summaries and note bodies are the record — the retrieval
design rests on quoting them rather than paraphrasing, and a machine-translated
note is exactly the failure this repo spends its time avoiding. A German caller
hearing an English note read back is correct.

What must translate is everything the agent adds around that: the sentence, the
figures, the dates, and the closed vocabularies.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from voice_agent.agent import locale
from voice_agent.agent.tools import db as database
from voice_agent.agent.tools import sales
from voice_agent.agent.tools.base import ago, money, spoken_date

REP = "Dani Brooks"
WHEN = datetime(2026, 5, 17, tzinfo=UTC)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return database.seed(tmp_path_factory.mktemp("data") / "wholesale.db")


# ───────────────────────────────────────────────────────────── the mechanism


def test_english_is_the_default() -> None:
    assert locale.current() == "en"


def test_use_restores_the_previous_language() -> None:
    with locale.use("de"):
        assert locale.current() == "de"
    assert locale.current() == "en"


@pytest.mark.parametrize(
    ("detected", "expected"),
    [("de", "de"), ("DE", "de"), ("de-DE", "de"), ("en", "en"), (None, "en")],
)
def test_recogniser_codes_map_onto_languages_we_can_speak(
    detected: str | None, expected: str
) -> None:
    assert locale.normalise(detected) == expected


def test_an_unsupported_language_falls_back_rather_than_half_translating() -> None:
    """Whisper is confident about languages this agent has no voice for.

    Answering in English is recoverable. Answering in fragments of three
    languages is not.
    """
    assert locale.normalise("fr") == "en"
    assert locale.normalise("zh") == "en"


def test_a_missing_message_raises_rather_than_leaking_its_key() -> None:
    """A key read aloud to a customer is worse than a crash in a test."""
    with pytest.raises(KeyError):
        locale.t("no_such_message")


def test_an_unknown_enum_value_passes_through() -> None:
    """A new segment in the data should read oddly, not end the call."""
    with locale.use("de"):
        assert locale.term("segment", "independent") == "unabhängiger"
        assert locale.term("segment", "franchise") == "franchise"


# ──────────────────────────────────────────────────────── numbers and dates


def test_german_thousands_separator_is_a_full_stop() -> None:
    """ "7,200" and "7.200" are the same number for different readers.

    A synthesiser reads the wrong one as a decimal, which turns seven thousand
    pounds into seven pounds twenty.
    """
    with locale.use("en"):
        assert money(7160) == "7,200 pounds"
    with locale.use("de"):
        assert money(7160) == "7.200 Pfund"


def test_german_decimals_use_a_comma() -> None:
    with locale.use("de"):
        assert money(1_250_000) == "1,2 Millionen Pfund"


def test_currency_stays_pounds() -> None:
    """The accounts are a UK merchant's. Converting would invent a rate."""
    with locale.use("de"):
        assert "Pfund" in money(500)
        assert "Euro" not in money(500)


def test_german_dates_are_written_the_german_way() -> None:
    with locale.use("en"):
        assert spoken_date(WHEN) == "the 17th of May"
    with locale.use("de"):
        assert spoken_date(WHEN) == "17. Mai"


def test_month_names_do_not_depend_on_the_host_locale() -> None:
    """``strftime('%B')`` returns whatever the machine is set to.

    That is English here and something else on a server in Frankfurt, which
    would make the output depend on where it happened to be deployed.
    """
    with locale.use("de"):
        assert spoken_date(datetime(2026, 3, 1, tzinfo=UTC)) == "1. März"
        assert spoken_date(datetime(2026, 12, 24, tzinfo=UTC)) == "24. Dezember"


def test_both_the_date_and_the_age_survive_translation() -> None:
    """The reason ``ago`` exists does not change with the language."""
    with locale.use("de"):
        said = ago(77, WHEN)
    assert "17. Mai" in said
    assert "77" in said


# ──────────────────────────────────────────────── chrome translates, data does not


def _brief(db_path: Path, language: str) -> str:
    tools = {spec.name: spec for spec in sales.build(db_path, REP)}
    with locale.use(language):  # type: ignore[arg-type]
        return asyncio.run(tools["brief_account"].handler(name="Marchwood"))


def test_the_sentence_around_the_data_is_german(db_path: Path) -> None:
    said = _brief(db_path, "de")
    assert "Ansprechpartner" in said
    assert "Im letzten Jahr" in said
    assert "account in" not in said


def test_account_names_are_not_translated(db_path: Path) -> None:
    """A company is called what it is called in every language."""
    assert "Marchwood" in _brief(db_path, "de")


def test_note_bodies_are_quoted_as_written(db_path: Path) -> None:
    """The whole retrieval design rests on quoting rather than paraphrasing.

    Notes are the record. Translating one would put words in a colleague's
    mouth, and the agent has no way to mark which parts it invented.
    """
    tools = {spec.name: spec for spec in sales.build(db_path, REP)}
    with database.connect(db_path, read_only=True) as db:
        body = str(
            db.execute(
                "SELECT n.body FROM notes n JOIN accounts a ON a.id = n.account_id"
                " JOIN reps r ON r.id = a.rep_id WHERE r.name = ? LIMIT 1",
                (REP,),
            ).fetchone()["body"]
        )
    with locale.use("de"):
        said = asyncio.run(tools["search_notes"].handler(question=body[:40]))
    assert body in said


def test_closed_vocabularies_do_translate(db_path: Path) -> None:
    """Unlike free text, an enum has a correct German word."""
    tools = {spec.name: spec for spec in sales.build(db_path, REP)}
    with locale.use("de"):
        said = asyncio.run(tools["get_recent_activity"].handler(name="Marchwood"))
    assert "einen Besuch" in said or "einen Anruf" in said or "eine E-Mail" in said
    assert " a visit " not in said
