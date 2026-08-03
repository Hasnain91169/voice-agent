"""Scenario definitions.

A scenario is a persona, a goal, and a set of checkable criteria. Deliberately
declarative and in YAML rather than in code, so adding a case is a data change —
the point of an eval suite is that it grows every time something goes wrong on a
real call, and that has to be cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

SCENARIO_DIR = Path(__file__).parent / "scenarios"


@dataclass(frozen=True)
class Scenario:
    """One simulated call."""

    name: str
    #: Who is calling and how they behave. Fed to the simulated caller.
    persona: str
    #: What the caller is trying to achieve.
    goal: str
    #: Opening line, so runs start identically.
    opening: str
    #: Maximum caller turns before the run is stopped.
    max_turns: int = 6

    #: Tools the agent should reach for at some point in the call.
    expects_tools: tuple[str, ...] = ()
    #: Substrings that should appear somewhere in the agent's speech. Each entry
    #: may offer alternatives separated by "|", because the system prompt asks
    #: for numbers written the way they are spoken.
    expects_facts: tuple[str, ...] = ()
    #: Things the agent must never say — commitments it cannot keep, or figures
    #: it has no source for.
    forbids: tuple[str, ...] = ()

    #: The agent must name at least one account the generator actually made
    #: behave this way. Asserting the archetype rather than a specific name
    #: keeps the scenario alive across a reseed.
    expects_account_archetype: str = ""
    #: The agent must not name any account belonging to another rep. This is
    #: the access boundary, checked against the data rather than hoped for.
    forbids_other_reps: bool = False
    #: The agent must name the category this account's peers buy and it does
    #: not, resolved from the data at check time.
    expects_category_gap_for: str = ""
    #: The caller's goal is one the agent should *refuse*. Without this, a
    #: correct refusal scores as a failed call — the judge is asked whether
    #: the caller got what they wanted, and here they should not.
    expects_refusal: bool = False

    #: Archetype of the account this call is about. When set, ``{account}`` in
    #: the opening, the goal and ``expects_category_gap_for`` is replaced with a
    #: real account of that kind, resolved from the data at run time.
    #:
    #: ``expects_account_archetype`` already kept the *assertions* alive across
    #: a reseed. The opening line did not: it named "Lowther Timber" directly,
    #: and a change to the seeding order renamed every account and quietly took
    #: that scenario to nought out of three. Half a scenario decoupled from the
    #: seed is not decoupled.
    about: str = ""
    #: Fill ``{account}`` with the account carrying this planted injection
    #: instead of one of a given archetype. Mutually exclusive with ``about``.
    about_injection: str = ""

    #: Faults to inject, e.g. {"asr_blank": 1} to blank the first transcript.
    faults: dict[str, Any] = field(default_factory=dict)
    #: Caller turn index at which to talk over the agent, if any.
    interrupt_at: int | None = None
    #: Require that the interruption actually cancelled the turn and that
    #: history recorded only what was heard. Audio mode only — text mode has
    #: no audio to talk over, so these scenarios are skipped rather than
    #: passed.
    expects_barge_in: bool = False
    #: Filename stem, so scenarios can be selected by their numeric prefix.
    source: str = ""

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "-")


def load(path: Path) -> Scenario:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Scenario(
        name=raw["name"],
        persona=raw["persona"].strip(),
        goal=raw["goal"].strip(),
        opening=raw["opening"].strip(),
        max_turns=int(raw.get("max_turns", 6)),
        expects_tools=tuple(raw.get("expects_tools", []) or []),
        expects_facts=tuple(raw.get("expects_facts", []) or []),
        forbids=tuple(raw.get("forbids", []) or []),
        faults=dict(raw.get("faults", {}) or {}),
        interrupt_at=raw.get("interrupt_at"),
        expects_account_archetype=raw.get("expects_account_archetype", ""),
        forbids_other_reps=bool(raw.get("forbids_other_reps", False)),
        expects_category_gap_for=raw.get("expects_category_gap_for", ""),
        expects_refusal=bool(raw.get("expects_refusal", False)),
        about=raw.get("about", ""),
        about_injection=raw.get("about_injection", ""),
        expects_barge_in=bool(raw.get("expects_barge_in", False)),
        source=path.stem,
    )


def load_all(directory: Path = SCENARIO_DIR) -> list[Scenario]:
    return [load(path) for path in sorted(directory.glob("*.yaml"))]


def resolve(scenario: Scenario, db_path: Path, rep: str) -> Scenario:
    """Fill ``{account}`` with a real account of the requested archetype.

    Raises rather than leaving the placeholder in place. A scenario that opens
    with "I am seeing brace account brace later" would run, fail every check,
    and look like an agent defect — which is worse than not running at all.
    """
    if not scenario.about and not scenario.about_injection:
        return scenario

    from evals import ground_truth

    if scenario.about_injection:
        found = ground_truth.injected_account(db_path, rep, scenario.about_injection)
        if found is None:
            raise LookupError(
                f"{scenario.name}: no {scenario.about_injection} payload planted on {rep}'s patch"
            )
        name = found
    else:
        candidates = ground_truth.accounts_with_archetype(db_path, rep, scenario.about)
        if not candidates:
            raise LookupError(
                f"{scenario.name}: no {scenario.about} account on {rep}'s patch to talk about"
            )
        # Sorted, so the same scenario names the same account every run.
        name = sorted(candidates.values())[0]

    def fill(text: str) -> str:
        return text.replace("{account}", name)

    return replace(
        scenario,
        opening=fill(scenario.opening),
        goal=fill(scenario.goal),
        expects_category_gap_for=fill(scenario.expects_category_gap_for),
    )
