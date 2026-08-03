"""Build the retrieval probe set: paraphrased questions with a known answer.

Run once, offline, and commit the result.

    uv run python scripts/generate_probes.py

A probe is a question a rep would actually ask, paired with the one note that
answers it. Recall against that pairing is the only way to tell whether a change
to the retriever helped, and without it "does search work?" collapses into
running two queries by hand and liking the look of the output — which is exactly
how a synonym that destroys precision gets shipped.

The generation prompt forbids reusing the note's distinctive words. That is the
whole point: a probe that shares its rare terms with the target measures nothing
except that BM25 can find an exact string. What has to be measured is the case a
rep creates every time they speak — asking about "complaints" when the note says
"frustrated", about "the new bloke" when the note names him.
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
OUT = Path("data/retrieval_probes.json")

#: Enough for a stable recall figure without a generation run that drags.
SAMPLE = 80

BRIEF = """Each numbered item below is a note a UK builders' merchant sales rep
typed after visiting a customer.

For each one, write the spoken question a busy rep would ask their voice
assistant that this note - and ideally only this note - would answer.

Rules:
- Write it the way someone says it out loud while driving. No punctuation
  flourishes, no formal phrasing.
- DO NOT reuse the note's distinctive words. If the note says "frustrated", ask
  about someone being unhappy or annoyed. If it names a person, ask about their
  role instead of their name. The question must be a paraphrase, not a quote.
- Keep the subject matter specific enough that the note is a plausible answer.
- Around 8 to 14 words.

Return a JSON array of strings, one question per note, in the same order, and
nothing else.

Notes:
{notes}"""


async def ask(client: anthropic.AsyncAnthropic, batch: list[str]) -> list[str]:
    listing = "\n".join(f"{i + 1}. {body}" for i, body in enumerate(batch))
    message = await client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": BRIEF.format(notes=listing)}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    start, end = text.find("["), text.rfind("]")
    if start < 0:
        raise SystemExit("model did not return a JSON array")
    questions = [str(q).strip() for q in json.loads(text[start : end + 1])]
    if len(questions) != len(batch):
        raise SystemExit(f"expected {len(batch)} questions, got {len(questions)}")
    return questions


async def main() -> int:
    settings = Settings()
    if settings.anthropic_api_key is None:
        print("VA_ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    path = database.seed(settings.db_path)
    with database.connect(path, read_only=True) as db:
        # Stratified by rep so the probe set is not concentrated on one patch,
        # and ordered deterministically so a rerun samples the same notes.
        rows = db.execute(
            "SELECT n.body, a.name AS account, r.name AS rep FROM notes n"
            " JOIN accounts a ON a.id = n.account_id"
            " JOIN reps r ON r.id = a.rep_id"
            " ORDER BY n.id"
        ).fetchall()

    step = max(1, len(rows) // SAMPLE)
    sample = [dict(row) for row in rows[::step]][:SAMPLE]
    print(f"sampling {len(sample)} of {len(rows)} notes")

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    batches = [sample[i : i + 20] for i in range(0, len(sample), 20)]
    results = await asyncio.gather(
        *(ask(client, [item["body"] for item in batch]) for batch in batches)
    )
    await client.close()

    probes = [
        {"question": question, "body": item["body"], "account": item["account"], "rep": item["rep"]}
        for batch, questions in zip(batches, results, strict=True)
        for item, question in zip(batch, questions, strict=True)
    ]

    OUT.write_text(
        json.dumps(
            {
                "generated_with": MODEL,
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "note": "Questions are paraphrases; distinctive terms are deliberately absent.",
                "probes": probes,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} — {len(probes)} probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
