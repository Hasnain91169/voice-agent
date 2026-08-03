"""Language for tool output, held per turn.

The agent answers in whatever language the caller just spoke, which is decided
per utterance by the speech recogniser. That creates a threading problem: the
functions that format money and dates sit four or five calls below the pipeline,
and passing a language argument through every one of them would touch every
signature in the tool layer for a value that never varies *within* a turn.

So it is a context variable, set once when the turn starts and read wherever
formatting happens. That is the case ``contextvars`` exists for, and it stays
testable because :func:`use` is an ordinary context manager.

**Why the tools translate and the model does not.** The system prompt forbids the
agent from restating figures in its own words — that rule is what stopped it
inventing dates and mis-rounding money, and it is load-bearing. Asking the model
to speak German from English tool output would require exactly the restatement
that rule forbids. Localising here keeps both properties: the tool does the
arithmetic and the wording, and the model repeats what it was given.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final, Literal

Language = Literal["en", "de"]

#: Languages with a full message catalogue. A detected language outside this set
#: falls back to English rather than emitting half-translated output, which is
#: worse to listen to than a consistent second language.
SUPPORTED: Final[frozenset[str]] = frozenset({"en", "de"})

DEFAULT: Final[Language] = "en"

_current: ContextVar[Language] = ContextVar("language", default=DEFAULT)


def current() -> Language:
    return _current.get()


def normalise(detected: str | None) -> Language:
    """Map a recogniser's language code onto one we can actually speak.

    Whisper returns two-letter codes and is confident about languages this agent
    has no voice, no prompt and no catalogue for. Anything unsupported becomes
    English, because answering in English is recoverable and answering in
    fragments of three languages is not.
    """
    if detected and detected.lower()[:2] in SUPPORTED:
        return detected.lower()[:2]  # type: ignore[return-value]
    return DEFAULT


@contextmanager
def use(language: Language) -> Iterator[None]:
    """Set the language for everything formatted inside this block."""
    token = _current.set(language)
    try:
        yield
    finally:
        _current.reset(token)


# ─────────────────────────────────────────────────────────── message catalogue

#: Keyed by message, then language. Kept as whole sentences rather than
#: assembled from fragments: word order differs between the two languages, and
#: a sentence stitched from translated pieces reads like neither.
_MESSAGES: Final[dict[str, dict[Language, str]]] = {
    # --- account lookup -------------------------------------------------
    "no_match_at_all": {
        "en": "There's nothing matching {name} on your patch.",
        "de": "Zu {name} gibt es in deinem Gebiet nichts.",
    },
    "no_match_but_close": {
        "en": (
            "I can't find {name} on your patch. The closest are {options}. Which one did you mean?"
        ),
        "de": (
            "Ich finde {name} in deinem Gebiet nicht. Am ähnlichsten sind {options}. "
            "Welchen meinst du?"
        ),
    },
    "and": {"en": "and", "de": "und"},
    # --- brief ----------------------------------------------------------
    "brief_header": {
        "en": "{name}, {segment} account in {region}. Your contact is {contact}.",
        "de": "{name}, {segment} Kunde in {region}. Dein Ansprechpartner ist {contact}.",
    },
    "brief_spend": {
        "en": "They've spent {amount} with us over the last year.",
        "de": "Im letzten Jahr haben sie {amount} bei uns gekauft.",
    },
    "brief_overdue": {
        "en": "They normally order every {days} days, and the last one was {when}.",
        "de": "Normalerweise bestellen sie alle {days} Tage, die letzte war {when}.",
    },
    "brief_on_time": {
        "en": "Last order was {when}, which is about normal for them.",
        "de": "Die letzte Bestellung war {when}, das ist für sie normal.",
    },
    "brief_trend_up": {
        "en": "Spend is up {pct} percent on the quarter before.",
        "de": "Der Umsatz liegt {pct} Prozent über dem Vorquartal.",
    },
    "brief_trend_down": {
        "en": "Spend is down {pct} percent on the quarter before.",
        "de": "Der Umsatz liegt {pct} Prozent unter dem Vorquartal.",
    },
    "brief_quiet": {
        "en": "Nobody's been in touch since {when}.",
        "de": "Seit {when} hat sich niemand gemeldet.",
    },
    "brief_signal": {"en": "Worth knowing: {signal}.", "de": "Wissenswert: {signal}."},
    "brief_gap": {
        "en": (
            "They don't buy {category} from us at all, and most accounts their size do. "
            "That's roughly {amount} a year."
        ),
        "de": (
            "{category} kaufen sie gar nicht bei uns, die meisten Kunden ihrer Größe schon. "
            "Das sind etwa {amount} im Jahr."
        ),
    },
    # --- risk -----------------------------------------------------------
    "risk_none": {
        "en": "Nothing on your patch is showing risk signals right now.",
        "de": "In deinem Gebiet zeigt gerade nichts Risikosignale.",
    },
    "risk_lead": {
        "en": "{count} accounts worth a look.",
        "de": "{count} Kunden lohnen einen Blick.",
    },
    "risk_lead_one": {
        "en": "1 account worth a look.",
        "de": "1 Kunde lohnt einen Blick.",
    },
    "reason_overdue": {
        "en": "usually orders about every {mean} days, but it has been {actual}",
        "de": "bestellt sonst etwa alle {mean} Tage, jetzt sind es {actual}",
    },
    "reason_trend": {
        "en": "spend is down {pct} percent on the previous quarter",
        "de": "der Umsatz liegt {pct} Prozent unter dem Vorquartal",
    },
    "reason_quiet": {
        "en": "nobody has been in touch since {when}",
        "de": "seit {when} hat sich niemand gemeldet",
    },
    # --- patch summary --------------------------------------------------
    "patch_empty": {
        "en": "There are no accounts assigned to your patch in this snapshot.",
        "de": "In diesem Datenstand sind deinem Gebiet keine Kunden zugeordnet.",
    },
    "patch_header": {
        "en": "Across your patch you have {count} {accounts}, with {amount} in the last year.",
        "de": "In deinem Gebiet hast du {count} Kunden mit {amount} im letzten Jahr.",
    },
    "patch_risk": {
        "en": "Risk is {high} high and {medium} medium, led by {names}.",
        "de": "Beim Risiko gibt es {high} hoch und {medium} mittel, vor allem {names}.",
    },
    "patch_risk_none": {
        "en": "Nothing is showing material churn risk right now.",
        "de": "Aktuell zeigt nichts ein klares Abwanderungsrisiko.",
    },
    "patch_growth": {
        "en": "Good news: {names} are growing.",
        "de": "Gute Nachricht: {names} wachsen.",
    },
    "patch_opportunities": {
        "en": "The clearest whitespace or intent is with {names}.",
        "de": "Die klarsten Luecken oder Kaufsignale liegen bei {names}.",
    },
    "patch_no_good_news": {
        "en": "No obvious growth or whitespace stands out from the summary.",
        "de": "Aus der Zusammenfassung sticht kein klares Wachstum und keine klare Luecke heraus.",
    },
    # --- opportunities --------------------------------------------------
    "opp_none": {
        "en": "Nothing obvious for {name}. They're buying across the range already.",
        "de": "Für {name} nichts Offensichtliches. Sie kaufen schon quer durch das Sortiment.",
    },
    "opp_gap": {
        "en": (
            "They buy no {category} from us, where {share} percent of similar accounts do. "
            "Worth about {amount} a year."
        ),
        "de": (
            "{category} kaufen sie nicht bei uns, {share} Prozent vergleichbarer Kunden schon. "
            "Etwa {amount} im Jahr wert."
        ),
    },
    "opp_signal": {"en": "They've also got {signal}.", "de": "Außerdem: {signal}."},
    # --- purchases ------------------------------------------------------
    "profile_none": {
        "en": "{name} hasn't ordered anything in the last year.",
        "de": "{name} hat im letzten Jahr nichts bestellt.",
    },
    "profile_lead": {
        "en": "Over the last year {name} bought {items}",
        "de": "Im letzten Jahr kaufte {name} {items}",
    },
    "profile_more": {
        "en": ", plus {count} smaller categories",
        "de": ", dazu {count} kleinere Warengruppen",
    },
    # --- closed vocabularies ---------------------------------------------
    #
    # Only the enums are here. Note bodies, activity summaries, contact names
    # and account names stay exactly as they were written, in whatever language
    # they were written in — they are the record, and the whole retrieval design
    # rests on quoting them rather than paraphrasing. A German caller hearing an
    # English note read back is correct behaviour; a German caller hearing a
    # machine-translated note is the failure this repo spends its time avoiding.
    #
    # The article is baked into the German values because it has to agree with
    # the noun, and a template cannot know whether the next word is masculine.
    "kind_visit": {"en": "a visit", "de": "einen Besuch"},
    "kind_call": {"en": "a call", "de": "einen Anruf"},
    "kind_email": {"en": "an email", "de": "eine E-Mail"},
    "segment_independent": {"en": "independent", "de": "unabhängiger"},
    "segment_regional": {"en": "regional", "de": "regionaler"},
    "segment_national": {"en": "national", "de": "überregionaler"},
    # --- activity and notes ---------------------------------------------
    "activity_none": {
        "en": "There's no contact history on that account at all.",
        "de": "Zu diesem Kunden gibt es überhaupt keine Kontakthistorie.",
    },
    "activity_you": {"en": "You", "de": "Du"},
    "activity_line": {
        "en": "{who} had {kind} with them on {when}, {summary}",
        "de": "{who} hattest am {when} {kind} mit ihnen, {summary}",
    },
    "note_yours": {"en": "Your", "de": "Deine"},
    "note_line": {
        "en": "{whose} note from {when} says: {body}",
        "de": "{whose} Notiz vom {when} lautet: {body}",
    },
    "notes_untrusted": {
        "en": "Unverified notes typed by reps, not system records.",
        "de": "Ungeprüfte Notizen von Außendienstlern, keine Systemdaten.",
    },
    "notes_none": {
        "en": (
            "Nothing in the notes{where} matches that. The notes only cover visits, "
            "so it may just never have been written down."
        ),
        "de": (
            "In den Notizen{where} passt dazu nichts. Die Notizen decken nur Besuche ab, "
            "vielleicht wurde es nie aufgeschrieben."
        ),
    },
    "notes_about": {"en": " about {name}", "de": " zu {name}"},
    "notes_lonely": {
        "en": (
            " That's the only note matching those words, so it's worth trying "
            "different wording before ruling it out."
        ),
        "de": (
            " Das ist die einzige Notiz zu diesen Wörtern, andere Formulierungen sind einen "
            "Versuch wert, bevor du es ausschließt."
        ),
    },
    # --- actions ---------------------------------------------------------
    "action_bad_kind": {
        "en": "I can only log a follow up or an escalation, not {kind}.",
        "de": "Ich kann nur eine Wiedervorlage oder eine Eskalation erfassen, nicht {kind}.",
    },
    "action_nothing_logged": {
        "en": " I've logged nothing for now.",
        "de": " Ich habe vorerst nichts erfasst.",
    },
    "action_escalated": {
        "en": "Flagged {name} for the account manager.",
        "de": "{name} für den Kundenbetreuer markiert.",
    },
    "action_logged": {
        "en": "Logged a follow up on {name}.",
        "de": "Wiedervorlage für {name} erfasst.",
    },
    "action_booked": {
        "en": "Follow up on {name} booked for {when}.",
        "de": "Wiedervorlage für {name} auf {when} gelegt.",
    },
}


def t(key: str, /, **fields: object) -> str:
    """One message, in the language of the current turn.

    Raises on an unknown key rather than returning the key itself. A missing
    translation that renders as ``brief_spend`` gets read aloud to a customer.
    """
    try:
        variants = _MESSAGES[key]
    except KeyError:  # pragma: no cover - a typo caught the first time it runs
        raise KeyError(f"no message named {key!r}") from None
    return variants.get(current(), variants[DEFAULT]).format(**fields)


def term(prefix: str, value: str, /) -> str:
    """Translate a closed vocabulary value, or return it untouched.

    Used for the handful of enum columns — activity kind, account segment.
    An unrecognised value passes straight through rather than raising, because
    a new segment appearing in the data should read oddly in German, not take
    the call down.
    """
    key = f"{prefix}_{value.lower().replace(' ', '_')}"
    if key in _MESSAGES:
        return t(key)
    return value
