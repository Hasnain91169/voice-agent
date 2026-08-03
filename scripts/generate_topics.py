"""Label the note corpus by topic, to measure broad search rather than lookup.

Run once, offline, and commit the result.

    uv run python scripts/generate_topics.py

``bench/retrieval.py`` already measured one retrieval task: given a question
paraphrasing one specific note, does that note come back? 79% recall@3. That is
the right measurement for *"what did they say about the Ackworth job"* and the
wrong one for *"has anyone mentioned a competitor"*, which is not a lookup at
all — it is a sweep, and it is answered badly if the retriever surfaces one of
the nine relevant notes and the agent concludes "no".

That second task went unmeasured, and a decision was made on the strength of the
first anyway: the synonym table was deleted because it did not help lookup. It
was never tested on sweeps, which is exactly where vocabulary mismatch bites — a
rep says "competitor" and the note says "Travis", "rival", "elsewhere",
"switching". This file supplies the labels that make sweeps measurable, so that
decision can be retaken on evidence instead of restated.

The questions are hand-written because they are the *input* — the thing a rep
actually says. The labels are the ground truth and come from the model, applied
to every note independently of any retriever, so no retriever can flatter itself.
"""

# ruff: noqa: ASYNC240  # a one-shot offline script; blocking file IO is correct here
from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from voice_agent.agent.tools import db as database
from voice_agent.config import Settings

MODEL = "claude-opus-5"
OUT = Path("data/topic_probes.json")

#: The sweeps a rep actually runs, each with the words they would say. Written by
#: hand on purpose: this is the input to the retriever, and generating it from
#: the corpus would leak the corpus vocabulary into the query, which is the one
#: thing that must not happen — it would measure a mismatch that never existed.
TOPICS: dict[str, str] = {
    "competitor": "has anyone mentioned a competitor or moving their business elsewhere",
    "unhappy": "which accounts are unhappy or complaining about something",
    "delivery": "any problems with deliveries or lead times",
    "pricing": "who has been pushing back on price",
    "credit": "anyone asking about credit terms or payment",
    "staff_change": "has anyone had a change of staff or a new buyer",
    "big_job": "who has a big job or tender coming up",
    "expansion": "anyone opening a new branch or expanding",
    "quiet": "which accounts have gone quiet or slowed down",
    "commitment": "what have we promised anyone that I need to follow up",
    "site_logistics": "anything about site access, timings or how they want deliveries",
    "personal": "anything personal worth remembering about a contact",
}

BRIEF = """You are labelling CRM visit notes written by a UK builders' merchant
sales rep, so that a search tool can be evaluated.

The topics are:
{topics}

Below are numbered notes. For each note, list every topic it is genuinely
about. A note counts for a topic if a rep asking that topic's question would
want to see this note - including when the note says the topic does NOT apply
("no complaints today" is about `unhappy`, because a rep sweeping for unhappy
accounts needs to see it and judge for themselves).

Be strict. A note that merely mentions a delivery in passing while being about
something else is not a `delivery` note. Most notes have one or two topics.
Some have none - that is a valid answer.

Return one line per note, in order, formatted exactly as:
<note number>: <comma-separated topic keys, or the word none>

No other text.

Notes:
{notes}"""

BATCH = 30


async def label(
    client: anthropic.AsyncAnthropic, batch: list[str], offset: int
) -> dict[int, list[str]]:
    listing = "\n".join(f"{i + 1}. {body}" for i, body in enumerate(batch))
    topics = "\n".join(f"- {key}: {question}" for key, question in TOPICS.items())
    async with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": BRIEF.format(topics=topics, notes=listing)}],
    ) as stream:
        message = await stream.get_final_message()
    text = "".join(block.text for block in message.content if block.type == "text")

    labels: dict[int, list[str]] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        number, _, rest = line.partition(":")
        try:
            index = int(number.strip().rstrip(".")) - 1
        except ValueError:
            continue
        if not 0 <= index < len(batch):
            continue
        keys = [k.strip() for k in rest.split(",") if k.strip() in TOPICS]
        labels[offset + index] = keys
    return labels


async def main() -> int:
    settings = Settings()
    if settings.anthropic_api_key is None:
        print("VA_ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    path = database.seed(settings.db_path)
    with database.connect(path, read_only=True) as db:
        rows = [
            dict(row)
            for row in db.execute(
                "SELECT n.body, r.name AS rep FROM notes n"
                " JOIN accounts a ON a.id = n.account_id"
                " JOIN reps r ON r.id = a.rep_id ORDER BY n.id"
            )
        ]
    print(f"labelling {len(rows)} notes against {len(TOPICS)} topics")

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    starts = list(range(0, len(rows), BATCH))
    results = await asyncio.gather(
        *(
            label(client, [r["body"] for r in rows[start : start + BATCH]], start)
            for start in starts
        )
    )
    await client.close()

    merged: dict[int, list[str]] = {}
    for part in results:
        merged.update(part)
    missing = [i for i in range(len(rows)) if i not in merged]
    if missing:
        print(f"  ! {len(missing)} notes went unlabelled", file=sys.stderr)

    # Inverted: topic -> the notes that belong to it, per rep. Retrieval is
    # rep-scoped, so relevance has to be too or the recall denominator counts
    # notes the search could never legitimately return.
    probes = []
    for key, question in TOPICS.items():
        by_rep: dict[str, list[str]] = {}
        for index, keys in merged.items():
            if key in keys:
                by_rep.setdefault(rows[index]["rep"], []).append(rows[index]["body"])
        probes.append({"topic": key, "question": question, "relevant": by_rep})
        total = sum(len(v) for v in by_rep.values())
        print(f"  {key:<16} {total:>3} notes")

    OUT.write_text(
        json.dumps(
            {
                "generated_with": MODEL,
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "note": "Questions hand-written; relevance labelled by the model, per note.",
                "probes": probes,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
