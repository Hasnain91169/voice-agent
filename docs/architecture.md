# How this works, and why

Written to be explained out loud. Roughly 12,000 lines across `src/`, `evals/`,
`bench/` and `tests/`. Read this once and you can answer questions about the
system without opening the source.

The through-line, if you only remember one thing: **the hard part of a voice
agent is not the model, it is the audio and the measurement.** Every genuinely
difficult decision in here is about who owns the microphone, what happens when
someone talks over you, and how you know whether any of it works.

---

## 1. The audio path

A phone call is a real-time system. Audio arrives every 20 milliseconds whether
you are ready or not, and if you fall behind you cannot catch up — you can only
drop.

```
caller ──audio──►  RxPump ── rechunk 10→20ms ── RMS ──► FrameChannel
                                                            │
                          ┌─────────────────────────────────┴──────┐
                    echo gate + VAD                          barge-in detector
                          │ utterance                              │ cancel
                          ▼                                        │
                         ASR ──► agent ──► clauses ──► TTS ──► playback ──► caller
```

**One task owns the inbound socket.** `rx.py` runs a single pump that reads the
transport, normalises endianness, rechunks 10ms slices into 20ms frames,
computes the RMS energy of each, and pushes `Frame` objects onto one channel.
Nothing else ever calls `transport.recv()`.

Everything downstream is a *consumer* of that channel: the voice-activity
detector reading for speech onset, and the barge-in detector reading for the
caller interrupting. They are the same subsystem asking one question — *is this
energy the caller, or my own voice coming back?*

The reason this matters: the previous implementation peeked at the socket from
inside the playback loop with a 0.1ms timeout, so playback and reception fought
over one reader. That is why its barge-in shipped disabled. With a single owner
the problem disappears rather than being worked around.

**Where the audio primitives live.** `audio/framing.py` (frame timing and the
rechunker), `audio/vad.py` (energy VAD, onset detection, hangover),
`audio/wav.py` (resampling, mono mixing, RMS — numpy, because `audioop` was
removed in Python 3.13).

## 2. A turn is a cancel scope

Within one turn, three things run as siblings: the LLM streaming tokens, the TTS
synthesising each clause as it becomes speakable, and playback pacing frames out
at 20ms. They live in one asyncio scope, so a barge-in cancels all three through
one mechanism.

The alternative — each unwinding separately — leaves the agent still
synthesising a sentence nobody will ever hear.

**Clause-level synthesis, not sentence-level.** Waiting for a full sentence
before synthesising costs most of a second. The clause assembler in `text.py`
emits at the first natural break, so TTS starts while the model is still talking.

## 3. Barge-in, and the part everyone gets wrong

Stopping the audio is the easy half. The hard half is what you *remember*.

Playback tracks bytes actually sent. When the caller interrupts, `SpokenTracker`
(`session.py`) maps those bytes back to the words that were actually heard, and
**that truncated text is what gets committed to conversation history** — not the
full generated response.

Commit the generated text instead and the agent spends the rest of the call
believing it told the caller things they never heard: referring back to them,
declining to repeat them, answering follow-ups that were never asked.

This is asserted, not asserted-about. The pipeline records generated characters
against spoken characters per turn, and the eval fails a barge-in scenario where
they match. Measured: **the agent stops speaking 219–328ms after being talked
over**, and history matched what the caller heard in 2 of 2 interrupted calls.

## 4. Failure paths are specified behaviour

The governing rule: **never dead air.** Silence makes a caller start talking,
which trips barge-in, which cancels the recovery, and the call spirals.

| Failure | Behaviour |
|---|---|
| ASR returns empty or low-confidence | Do not advance the turn, do not feed a placeholder to the model. Play a clarifier. After two, check the line. |
| No token by deadline (~1.5s) | Play a holding line to cover the gap, keep the stream alive |
| Model dies mid-stream | Flush buffered text if it ends at a clause boundary, else speak a recovery line |
| TTS fails on one clause | Clauses are independent: retry, fall back, or skip. A missing clause beats a dropped call |
| Tool errors | Return a structured error *into* the graph so the agent recovers conversationally rather than raising |
| Provider down at startup | Fail the health check, not the first live call |

Every fixed line is pre-synthesised at startup (`prompts_cache.py`), so no
recovery path pays synthesis latency at the exact moment it is needed.

## 5. The agent layer, and where the framework stops

LangGraph owns reasoning: `agent/graph.py` is `ingest → agent → (tools ⇄ agent)`,
streaming tokens out of the agent node. `agent/runner.py` adapts it to a
`TurnSource` protocol so the pipeline never branches on which backend is behind
it.

**The framework never sits inside the 20ms frame loop.** That boundary is
deliberate and it is why barge-in can cancel cleanly — the cancel scope is plain
asyncio, not a framework's execution model.

**State ownership is one-directional.** LangGraph state is authoritative for
everything the agent reasons over. `Session` is authoritative for transport and
turn control only — thresholds, timers, playback handle, metrics. `Session` holds
a `thread_id` string, never a copy of the conversation. There is exactly one
crossing point, at the end of a turn, in one direction.

## 6. The domain

A sales assistant for a builders' merchant, speaking to a rep who is driving.
Three data sources a real distributor would have separately — ERP orders, CRM
contact history, website behaviour — because the useful signals only exist where
they cross. An account that stopped ordering is a fact in one table; an account
that stopped ordering *and* has been browsing a category it never buys is a
reason to call.

**The derived layer is the product** (`agent/insights.py`). Reorder cadence
measured against each account's *own* rhythm, because an account that orders
weekly and one that orders quarterly are not both late at sixty days. Revenue
trend. Category gaps against a size-matched peer cohort. A churn score that
carries the reasons that produced it — the score is a sort key, the reasons are
the output, because "68" is useless to someone driving.

**Three guardrails live in code, not in the prompt**, because a prompt is a
request and a function is a fact:

- **Peers are never named.** Gap analysis returns "most accounts your size buy
  this" and cannot return "Northgate buys this" — the aggregate is all the
  function computes.
- **Prices are never quoted.** There is no pricing tool to call.
- **Account lookup never leaves the rep's book.** Every query joins through the
  rep. An earlier version fell back to a global search when the scoped match
  missed; that forgave the access boundary in the name of forgiving typos.

**Tool output is written to be spoken.** Figures are pre-rounded and dates are
pre-resolved, so the model has no arithmetic left to do — it was getting it
wrong. `money(7160)` returns "7,200 pounds", not "7,160.00". `ago()` returns both
the date and the age, because given only "77 days" the model computes a date
against its own idea of today, which is its training cutoff rather than the
dataset's frozen snapshot.

## 7. Retrieval

Everything above is computed from tables. Visit notes are prose, and hold what no
schema has a column for.

BM25 over SQLite FTS5 — at a few hundred short notes per rep, an embedding index
is a heavier dependency for a corpus that fits in a CPU cache.

**Two different jobs, measured separately**, and separating them was the whole
lesson:

- **Lookup** — *"what did they say about the Ackworth job?"* One note answers it.
- **Sweep** — *"has anyone mentioned a competitor?"* Nine notes answer it, they
  use nine different words, and there are three slots.

| | lookup recall@3 | sweep hit@3 | sweep precision@3 |
|---|---:|---:|---:|
| terms + synonyms (shipped) | 72% | **73%** | **35%** |
| terms only | **75%** | 54% | 26% |
| AND instead of OR | 1% | – | – |

`AND` returns nothing at all — a spoken question always carries one word the
corpus lacks, and one such word empties an `AND`.

**BM25 cannot see negation.** "Who is unhappy" retrieves "not one complaint
today". Nothing in the retrieval layer fixes that; what contains it is returning
notes verbatim with author and date, so the model reads the "not" the index
could not.

## 8. Evaluation

Two kinds of check, and the split is the point.

**Deterministic** — was the expected tool called, did the agent name an account
that is genuinely at risk, did it say a forbidden phrase, did it name another
rep's customer. Exact, cheap, never disagrees with itself.

**A model judge** — for the two questions that cannot be pattern-matched: did the
agent behave correctly, and did it assert anything its tools did not support.

**Ground truth comes from the generator, not from hand-written expectations.**
The seeder records which accounts it made behave which way in a `seed_truth`
table no tool can read. A scenario asserts "the agent named an account that is
genuinely at risk", so re-tuning the seed does not invalidate it.

**Two run modes.** Text mode drives the agent layer directly — fast, suitable
for CI. Audio mode runs the real pipeline over a loopback transport: caller turns
synthesised and fed in as 20ms frames, agent audio captured coming out. It
exercises the VAD, endpointer, echo gate and barge-in, and produces the only
honest first-audio measurement in the repository, timed from outside the process.

**Scores are rates over N runs.** One run of an LLM-judged suite is a sample. Five
consecutive runs of an earlier version gave 7/7 four times and 6/7 once, with a
*different* scenario failing each time.

**Adversarial input is part of the suite.** Notes are free text the agent reads
aloud, so anything written in them arrives in the model's context looking like
data. Three prompt-injection payloads are planted on every rep's patch.

## 9. The measurement failures — the part worth leading with

Building an eval harness is common. Knowing when yours is lying to you is not.
Every one of these looked like a working measurement:

- A judge returning malformed JSON left the verdict fields unset, which the pass
  logic read as "not checked". **A scenario whose measurement had broken scored
  green.**
- The grounding rubric asked whether a claim *appeared in* the tool output —
  string overlap — while the system prompt orders the agent to round figures for
  speech. It failed "mid-May" against "the 17th of May".
- The leak guardrail matched the first two words of an account name, so briefing
  your own "Severn Valley Industrial Fasteners" registered as naming another
  rep's "Severn Valley Timber & Board". **A guardrail that cries wolf trains you
  to skim past the one time it means it.**
- The retrieval bench reported a confident **0% at every depth** after the corpus
  was regenerated and the probes were left pointing at notes that no longer
  existed.
- **First-audio latency was computed from the transport's *last* frame**, so it
  measured time-to-end-of-answer and published it as responsiveness. It also went
  negative when the agent spoke over the caller, and the summary printed
  `p50 -3781ms — within budget`.
- A barge-in limitation was **published in the README as a flaw in the system
  when it was a flaw in the test** — the summary counted interruptions *injected*
  rather than turns actually cancelled, a number an agent finishing normally
  produces identically.
- A prompt change made for good reasons let an injected note reach a rep as
  authorisation, **with all 212 unit tests green**.

That last one is the argument for adversarial simulation in a single example.

## 10. Numbers you can defend

| | |
|---|---|
| Barge-in, caller talking over to last frame | **219–328ms** |
| First audio p50, external, at the shipped endpointer | **719ms** (budget 1000ms) |
| First audio p95 | 1250–1360ms — **over** the 1200ms target |
| Agent spoke before the caller finished | 4 of 43 turns (was 10 of 30 at the old setting) |
| Retrieval, two tasks | 72% lookup recall@3, 73% sweep hit@3, 0.2ms |
| Behaviour suite | 25 of 33 scenario-runs over three runs |
| Tests | 212, ruff and mypy clean |

**The endpointer story is the best single answer to "tell me about a tradeoff".**
`stop_hang_ms` was 250ms, derived by working backwards from an 800ms latency
target. Only the latency half was ever measured. Swept against real audio:

| `stop_hang_ms` | first audio p50 | cut the caller off |
|---:|---:|---:|
| 250 | 624ms | 10 of 30 turns |
| **450 (shipped)** | **938ms** | 2 of 32 |
| 650 | 1250ms | 1 of 34 |

Talking over the caller on a third of turns is not a latency win. The published
budget moved 800ms → 1000ms to fit the endpointer the system can actually run.

## 11. Stack

| Layer | Choice | Why |
|---|---|---|
| ASR | faster-whisper `small.en`, CUDA | 84ms warm on GPU; 1504ms on CPU — the profile split is GPU vs CPU, not local vs cloud |
| TTS | Piper, resident process | 89–123ms; spawning per clause costs 466ms, ~340ms of it process start paid every clause |
| LLM | Ollama local, Anthropic cloud | qwen2.5:7b is 10× faster and gets 1 tool call in 3; Haiku matches Sonnet on tools at a third the price |
| Agent | LangGraph | reasoning only, never in the frame loop |
| Retrieval | SQLite FTS5, BM25 | no extra dependency for a corpus this size |
| Data | SQLite, frozen snapshot | deterministic seed so eval assertions survive a reseed |
| Config | pydantic-settings | the latency budget is *data*, so bench and evals assert against the same numbers the README publishes |
| Transport | protocol + browser / loopback / (Vonage) | the eval runs the real pipeline because loopback satisfies the same protocol |

**Warm-up is load-bearing.** First CUDA inference 9.26s against 50ms warm; Ollama
4389ms cold against 98ms warm. `/health` returns 503 until warm, deliberately.

---

## Glossary

**VAD** — voice activity detection. Deciding, frame by frame, whether the caller
is speaking. Here it is energy-based against a noise floor calibrated at call
start, not a neural model.

**Endpointing** — deciding the caller has *finished*, not just paused. The single
biggest latency lever, and the one that trades directly against interrupting
people mid-sentence.

**Hangover** — how much silence to wait through before calling it the end of a
turn. 450ms here, chosen by measurement.

**Barge-in** — the caller talking over the agent, and the agent stopping. Real
barge-in also fixes what it remembers saying.

**Echo gate** — not mistaking your own voice, returning down the line, for the
caller speaking.

**TTFT** — time to first token. How long the model takes to start answering.

**First-audio latency** — from the caller falling silent to the first audio byte
leaving the server. The number a caller actually experiences as responsiveness.

**BM25** — the standard keyword relevance ranking. Scores documents on term
frequency, weighted so rare words count for more. No notion of meaning, and
none of negation.

**recall@k** — of the notes that *should* have been found, how many appeared in
the top k results.

**Prompt injection** — instructions hidden in data the model reads, hoping it
treats them as commands. Here: text pasted into a CRM note that the agent
retrieves and reads aloud.

**Grounding** — whether what the agent said is supported by what its tools
returned. Judged as entailment, not string overlap, because the agent is
instructed to round figures for speech.
