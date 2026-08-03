# Design assets

First batch of GitHub and slide-deck visuals for `voice-agent`.

The source of truth is the editable SVG/HTML/Mermaid file beside each PNG. PNGs are export-ready previews for GitHub, slide decks, and social cards.

## Ground rules

- Treat the project as in progress.
- Use README terminology and measured numbers exactly.
- Do not present unfinished work as shipped: Vonage telephony is deliberately not implemented; Deepgram and ElevenLabs adapters are absent; German support is built but needs multilingual ASR and a German voice configured for a local bilingual run.
- Label known weaknesses rather than hiding them: first-audio p95 is over the 1200ms target, behavioural score moves between runs, and judge conflation around `goal_met` remains unresolved.
- Keep visuals native to the current app style: dark console, mint signal color, waveform accents, quiet panels, and precise technical labels.

## Assets

| Asset | PNG export | Intended use |
|---|---|---|
| [Repository hero/banner](repo-hero-banner.svg) | [repo-hero-banner.png](repo-hero-banner.png) | README top banner and GitHub social preview. |
| [LangGraph tool-use flow](langgraph-tool-use.svg) | - | README visual tour and agent/tool-use explanation. |
| [Business insight layer](data-insight-layer.svg) | - | README visual tour and domain/data explanation. |
| [Full real-time audio architecture](realtime-audio-architecture.svg) | [realtime-audio-architecture.png](realtime-audio-architecture.png) | README architecture section and technical slide. |
| [Turn/cancel-scope timeline](turn-cancel-scope-timeline.svg) | [turn-cancel-scope-timeline.png](turn-cancel-scope-timeline.png) | Slide explaining latency budget and cancellation boundary. |
| [Barge-in and spoken-memory visual](barge-in-spoken-memory.svg) | [barge-in-spoken-memory.png](barge-in-spoken-memory.png) | README commitment section and interview slide. |
| [Endpointing latency-versus-interruption chart](endpointing-tradeoff.svg) | [endpointing-tradeoff.png](endpointing-tradeoff.png) | Trade-off slide and README measurement section. |
| [Evaluation architecture/dashboard](evaluation-architecture-dashboard.svg) | [evaluation-architecture-dashboard.png](evaluation-architecture-dashboard.png) | Eval section visual and slide-deck dashboard frame. |
| [Measurement-failures visual](measurement-failures.svg) | [measurement-failures.png](measurement-failures.png) | Slide opener for eval rigor; README measurement-failures section. |
| [Slide-deck storyboard/style guide](slide-deck-storyboard-style-guide.html) | - | Deck structure, palette, copy rules, and slide sequencing. |
| [Architecture Mermaid](realtime-audio-architecture.mmd) | - | Lightweight editable diagram for README/wiki variants. |
| [Evaluation Mermaid](evaluation-architecture.mmd) | - | Lightweight editable eval-flow diagram. |

## Measured claims represented

- External first audio: p50 719ms across 43 turns.
- Endpointer sweep: 250ms -> 624ms p50 and 10 of 30 overlaps; 450ms shipped -> 938ms p50 and 2 of 32 overlaps; 650ms -> 1250ms p50 and 1 of 34 overlaps.
- Budget components: 450 + 50 + 200 + 80 + 150 + 70 = 1000ms.
- Barge-in: 2 interruptions injected, 3 turns actually cancelled by the pipeline, p50 328ms from being talked over to the last frame, and history matched heard text in 2 of 2 interrupted calls.
- Behaviour suite: 25 of 33 scenario-runs over three runs, with skipped audio rows excluded.
- Tests: 279.

## Recommended placement

- `repo-hero-banner`: top of README.
- `realtime-audio-architecture`: after "The audio path".
- `turn-cancel-scope-timeline`: after "A turn is a cancel scope".
- `barge-in-spoken-memory`: after the spoken-memory commitment.
- `endpointing-tradeoff`: in "Turn-taking" or a deck trade-off slide.
- `evaluation-architecture-dashboard`: in "Behaviour" or "Evaluation".
- `measurement-failures`: lead slide for evaluation rigor or near "Things the harness got wrong".
