"""Generate the free-text visit notes the retrieval tool searches over.

Run once, offline, and commit the result. Seeding then stays deterministic and
needs no API key, which is the property that lets the whole repo run with none.

    uv run python scripts/generate_notes.py

**Why this exists.** The first version of the seeder chose each note from nine
hard-coded sentences, so 267 notes contained nine distinct strings. Retrieval
over that corpus cannot be evaluated: every query matches an exact duplicate,
any implementation scores perfectly, and the demo proves nothing except that
the author did not look at their own data. A retrieval corpus has to contain
things that are actually different from each other, phrased the way different
people phrase them — near-misses, partial paraphrases, and specific facts that
appear exactly once.

Notes are written to slots rather than to accounts, so the same corpus can be
reused across reseeds and the generator stays free to assign them. The archetype
pools exist because a note has to be consistent with the numbers: an account the
generator made quiet on purpose should not have a note saying business is
booming, or the eval's ground truth and its text disagree.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from voice_agent.config import Settings

MODEL = "claude-opus-5"
OUT = Path("data/notes_corpus.json")

#: Slots the seeder fills per account. Keeping them explicit means a generated
#: note can be checked for placeholders the seeder cannot fill.
SLOTS = "{contact}, {category}, {competitor}"

BRIEF = """You are writing the free-text CRM notes a field sales rep for a UK
builders' merchant and industrial supplies wholesaler types into their phone
after visiting a trade customer.

Write {count} notes for an account whose situation is: {situation}

Rules:
- One to three sentences. Written fast, in a hurry, on a phone. Plain UK trade
  English. No markdown, no bullet points, no emoji.
- Every note must be genuinely DIFFERENT from the others - different topic,
  different phrasing, different detail. Do not write variations of one note.
- Roughly half should contain a specific, concrete, retrievable fact: a named
  person and their role, a job or site they are working on, a product line, a
  timescale, a quantity, a piece of kit, something personal the rep noted to
  remember. Invent plausible UK names and places.
- The rest can be softer: impressions, rapport, things to watch.
- You may use these placeholders where they read naturally, and only these:
  {slots}. Use them sparingly - most notes should not contain any.
- Do not number them. Do not add commentary.

Return one note per line, nothing else. No numbering, no bullets, no
surrounding quotes, no blank lines between them."""

SITUATIONS = {
    "generic": "ordinary and stable, nothing unusual happening",
    "healthy": "a good account, buying well, happy with service, growing steadily",
    "declining": (
        "spending less than they used to, work drying up, some frustration "
        "creeping in but not lost yet"
    ),
    "at_risk": (
        "gone quiet, not ordering, actively unhappy or being courted by a "
        "competitor, at real risk of being lost"
    ),
    "category_gap": (
        "buying well in some lines but conspicuously not in others, sourcing "
        "part of their range elsewhere out of habit rather than grievance"
    ),
    "intent_spike": (
        "showing sudden new interest - asking for quotes, pricing up a big "
        "upcoming job, planning an expansion"
    ),
}

#: Sized per pool to actual demand rather than uniformly. The seeder draws
#: without replacement across the WHOLE database, and the archetypes are not
#: evenly distributed: 22 of 40 accounts are ``healthy`` and need roughly 150
#: notes between them, where ``intent_spike`` covers 3 and needs 25. A flat 70
#: per pool drained ``healthy`` and ``generic`` and fell back to reusing notes —
#: the exact duplication this file exists to prevent, reintroduced quietly.
#:
#: Undersized pools are also what put "Kev their driver" on two different
#: companies: the notes carry specific invented people, so reusing one across
#: accounts is visibly wrong to anyone who reads the data.
POOL_SIZES = {
    "generic": 90,
    "healthy": 170,
    "declining": 55,
    "at_risk": 45,
    "category_gap": 50,
    "intent_spike": 35,
}

#: One batch per request caps out well before 170 notes, so large pools are
#: asked for in chunks and merged.
BATCH = 45


async def _one_batch(
    client: anthropic.AsyncAnthropic, pool: str, situation: str, count: int
) -> list[str]:
    # Streamed because the budget is large enough that the SDK refuses a
    # non-streaming request outright: a batch of 45 notes plus adaptive thinking
    # can run past the ten-minute non-streaming ceiling.
    async with client.messages.stream(
        model=MODEL,
        max_tokens=32000,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": BRIEF.format(count=count, situation=situation, slots=SLOTS),
            }
        ],
    ) as stream:
        message = await stream.get_final_message()
    text = "".join(block.text for block in message.content if block.type == "text")
    # Line-delimited rather than JSON. The notes are full of apostrophes and
    # quoted speech, and a single unescaped quote invalidates a whole batch —
    # which is exactly what happened. A line cannot be broken by its contents.
    notes = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        # Short lines are headers or stray commentary, never a visit note.
        if len(cleaned) > 25:
            notes.append(cleaned)
    if not notes:
        raise SystemExit(f"{pool}: model returned nothing usable")
    return notes


async def generate(
    client: anthropic.AsyncAnthropic, pool: str, situation: str, wanted: int
) -> list[str]:
    sizes = [BATCH] * (wanted // BATCH) + ([wanted % BATCH] if wanted % BATCH else [])
    batches = await asyncio.gather(*(_one_batch(client, pool, situation, size) for size in sizes))
    unique = list(dict.fromkeys(note for batch in batches for note in batch))
    if len(unique) < wanted * 0.8:
        print(f"  ! {pool}: only {len(unique)} distinct, wanted {wanted}", file=sys.stderr)
    return unique


async def main() -> int:
    settings = Settings()
    if settings.anthropic_api_key is None:
        print("VA_ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    pools: dict[str, list[str]] = {}
    results = await asyncio.gather(
        *(
            generate(client, pool, situation, POOL_SIZES[pool])
            for pool, situation in SITUATIONS.items()
        )
    )
    for pool, notes in zip(SITUATIONS, results, strict=True):
        pools[pool] = notes
        print(f"  {pool:<14} {len(notes)} notes")

    await client.close()
    _write(pools)
    total = sum(len(v) for v in pools.values())
    print(f"wrote {OUT} — {total} notes across {len(pools)} pools")
    return 0


def _write(pools: dict[str, list[str]]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_with": MODEL,
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "note": "Generated once and committed. Seeding is offline and deterministic.",
                "pools": pools,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
