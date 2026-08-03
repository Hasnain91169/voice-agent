"""Measure the note retriever against two different jobs.

    uv run python -m bench.retrieval

They are genuinely different problems, and conflating them is how a wrong
decision gets made with a number attached to it.

**Lookup** — *"what did they say about the Ackworth job?"* One note answers it.
The probe set in ``data/retrieval_probes.json`` pairs a paraphrased question
with that note, and recall@k is whether it comes back. The paraphrases avoid the
target's distinctive vocabulary on purpose, so this measures the case a speaking
rep creates rather than the case an exact-match index is good at.

**Sweep** — *"has anyone mentioned a competitor?"* Nine notes answer it, spread
across the patch, and the retriever gets three slots. ``data/topic_probes.json``
labels every note against every topic, so coverage is measurable: how much of
what exists actually surfaced.

The distinction is not academic. Query expansion was measured on lookup, found
to be worth nothing, and deleted — and the deletion was then assumed to hold for
sweeps, which had never been measured at all. The agent went on to search the
single word "competitor", surface one note out of nine, and tell a rep "not
recently, no". Expansion is back, because on sweeps it is worth nineteen points
of hit rate against two points of lookup recall. Both tables print here now so
that trade stays visible rather than being rediscovered.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from voice_agent.agent import retrieval
from voice_agent.agent.tools import db as database

PROBES = Path("data/retrieval_probes.json")
TOPICS = Path("data/topic_probes.json")

#: Deeper than the tool returns, so the report distinguishes a ranking problem
#: (present at 10, missing at 3) from a matching problem (absent entirely).
DEPTHS = (1, 3, 10)

#: What the tool actually hands the model. Precision here is what the caller
#: hears, so it is worth more than coverage further down the list.
SPOKEN = 3


@dataclass
class Lookup:
    name: str
    recalls: dict[int, float]
    no_results: int
    median_ms: float


@dataclass
class Sweep:
    name: str
    hit_rate: float
    precision: float
    coverage: float
    median_ms: float


def measure_lookup(name: str, probes: list[dict[str, str]], **options: object) -> Lookup:
    hits = dict.fromkeys(DEPTHS, 0)
    empty = 0
    timings: list[float] = []

    with database.connect(database.DEFAULT_PATH, read_only=True) as db:
        for probe in probes:
            started = time.perf_counter()
            found = retrieval.search(
                db, probe["question"], probe["rep"], limit=max(DEPTHS), **options
            )
            timings.append((time.perf_counter() - started) * 1000)
            if not found:
                empty += 1
                continue
            bodies = [note.body for note in found]
            for depth in DEPTHS:
                if probe["body"] in bodies[:depth]:
                    hits[depth] += 1

    return Lookup(
        name=name,
        recalls={k: hits[k] / len(probes) for k in DEPTHS},
        no_results=empty,
        median_ms=statistics.median(timings),
    )


def measure_sweep(name: str, probes: list[dict], **options: object) -> Sweep:
    """How much of what exists surfaces, and how much of what surfaces is real.

    Three numbers, because a sweep can fail in three ways. ``hit_rate`` is
    whether the rep gets any true answer at all. ``precision`` is how much of
    what they hear is actually on topic — the cost of widening a query.
    ``coverage`` is the share of the relevant notes reachable at depth 10, which
    is the ceiling a better ranker could deliver without a better matcher.
    """
    asked = 0
    hits = 0
    precisions: list[float] = []
    coverages: list[float] = []
    timings: list[float] = []

    with database.connect(database.DEFAULT_PATH, read_only=True) as db:
        for probe in probes:
            for rep, relevant in probe["relevant"].items():
                if not relevant:
                    continue
                asked += 1
                wanted = set(relevant)
                started = time.perf_counter()
                found = retrieval.search(db, probe["question"], rep, limit=max(DEPTHS), **options)
                timings.append((time.perf_counter() - started) * 1000)

                spoken = [n.body for n in found[:SPOKEN]]
                deep = {n.body for n in found}
                if wanted & set(spoken):
                    hits += 1
                if spoken:
                    precisions.append(len(wanted & set(spoken)) / len(spoken))
                coverages.append(len(wanted & deep) / len(wanted))

    return Sweep(
        name=name,
        hit_rate=hits / asked if asked else 0.0,
        precision=statistics.mean(precisions) if precisions else 0.0,
        coverage=statistics.mean(coverages) if coverages else 0.0,
        median_ms=statistics.median(timings) if timings else 0.0,
    )


def _stale(probes: list[dict[str, str]]) -> str | None:
    """Refuse to score probes that point at notes the corpus no longer has.

    Regenerating the note corpus orphans every probe, and the bench then
    reports a confident 0% at every depth — which reads as a catastrophic
    retrieval regression rather than as "you changed the data and forgot the
    probes". A measurement that cannot detect its own inputs being invalid is
    worse than no measurement, because it is believed.
    """
    with database.connect(database.DEFAULT_PATH, read_only=True) as db:
        corpus = {str(row["body"]) for row in db.execute("SELECT body FROM notes")}
    missing = [p for p in probes if p["body"] not in corpus]
    if not missing:
        return None
    return (
        f"{len(missing)} of {len(probes)} probes reference notes that are not in the "
        f"corpus — the notes were regenerated after the probes were built.\n"
        f"Rebuild them: uv run python scripts/generate_probes.py"
    )


def main() -> int:
    for path in (PROBES, TOPICS):
        if not path.exists():
            print(f"{path} is missing — run the matching script in scripts/")
            return 2
    database.seed(database.DEFAULT_PATH)
    probes = json.loads(PROBES.read_text(encoding="utf-8"))["probes"]
    topics = json.loads(TOPICS.read_text(encoding="utf-8"))["probes"]
    if (stale := _stale(probes)) is not None:
        print(stale)
        return 2

    print("# Retrieval\n")
    print(f"## Lookup — find the one note that answers this ({len(probes)} probes)\n")
    lookups = [
        measure_lookup("terms + synonyms (shipped)", probes),
        measure_lookup("terms only", probes, expand=False),
        measure_lookup("AND instead of OR", probes, conjunction="AND"),
    ]
    print("| Configuration | " + " | ".join(f"recall@{k}" for k in DEPTHS) + " | empty | median |")
    print("|---|" + "--:|" * (len(DEPTHS) + 2))
    for row in lookups:
        cells = " | ".join(f"{row.recalls[k]:.0%}" for k in DEPTHS)
        print(f"| {row.name} | {cells} | {row.no_results} | {row.median_ms:.2f}ms |")

    labelled = sum(len(v) for p in topics for v in p["relevant"].values())
    print(
        f"\n## Sweep — find every note about a topic ({len(topics)} topics, {labelled} labelled)\n"
    )
    sweeps = [
        measure_sweep("terms + synonyms (shipped)", topics),
        measure_sweep("terms only", topics, expand=False),
    ]
    print(f"| Configuration | hit@{SPOKEN} | precision@{SPOKEN} | coverage@10 | median |")
    print("|---|--:|--:|--:|--:|")
    for row in sweeps:
        print(
            f"| {row.name} | {row.hit_rate:.0%} | {row.precision:.0%} | "
            f"{row.coverage:.0%} | {row.median_ms:.2f}ms |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
