# README artifact checklist

Use the README as a complete application packet: a reviewer should be able to understand the system, trust the evidence, watch the behaviour, and know what is intentionally not deployed.

## Recommended artifacts

| Artifact | Purpose | Suggested README placement |
|---|---|---|
| 90-second demo video | Shows the browser call, barge-in, tool calls, and eval dashboard without requiring setup | Top section, after the one-line pitch |
| Slide deck | Gives a structured project walkthrough before the live demo | Link beside the video |
| Architecture diagram | Makes socket ownership, VAD, ASR, LangGraph, TTS, playback, and cancel scope obvious | "The audio path" |
| Evaluation report | Shows scenario count, pass rate, failures, command, model, and date | "Measured" or "Evaluation" |
| Sample transcripts | Lets reviewers inspect successful and failed scenarios quickly | Collapsible section after evaluation |
| Implemented/planned matrix | Prevents overclaiming around Vonage, Deepgram, ElevenLabs, German, Docker, and Vercel | "What is built" before "What is not built" |
| CI badge | Makes pytest, ruff, and mypy credibility visible | README header |
| TMC production note | Separates production telephony evidence from this local/browser architecture project | "Production context" |

## Suggested positioning

This repo should not apologise for being browser-first. Position it as:

> This project is deliberately browser/local so the hard voice-agent behaviours are reproducible without a paid phone number: frame ownership, endpointing, barge-in, spoken memory, tool use, and audio-loopback evaluation. Production telephony evidence comes from my separate TMC project, where I built a Vonage voice agent from scratch and shipped it to production.

## Vercel note

Use Vercel for a hosted showcase page, README landing page, or cloud-provider variant. Be careful about presenting the current local stack as a normal Vercel deployment: the repo depends on long-lived media sockets, warm ASR/TTS/LLM providers, local model assets, and optional GPU acceleration. A clean hosted artifact could be:

- Static project page with video, deck, architecture, eval summary, and setup commands.
- Hosted dashboard that reads the committed `eval-summary.json`.
- Optional cloud-only branch later, using hosted ASR/TTS/LLM providers and no local model binaries.

## Demo video outline

1. Open with the problem: voice agents fail at audio timing, interruption, and measurement.
2. Show the browser call starting after provider warm-up.
3. Ask which accounts are slipping and point out tool calls plus grounded figures.
4. Interrupt the agent mid-answer and show cancellation.
5. Ask about a poisoned note and show the untrusted-note refusal.
6. End on the eval summary, known failures, and why they are useful.

## Quick README fixes

- Update the test count from 212 to 268.
- Clarify German as code-supported/localisation-ready but not default-demo shipped unless multilingual ASR and German Piper voice assets are installed.
- Clarify low-latency profile provider adapters as planned unless Deepgram, ElevenLabs, and OpenAI adapters are implemented.
- Add a short "production telephony" note referencing the TMC Vonage project.
