"""Static checks for the browser demo.

The live audio path is mostly exercised in a browser, but a few regressions are
cheap to catch from the checked-in HTML. In particular, playback should stay in
the AudioWorklet queue rather than scheduling one AudioBufferSource per 20ms
server frame; that path was prone to audible gaps where a long answer played the
first burst and then appeared to resume only after more user audio arrived.
"""

from __future__ import annotations

from pathlib import Path

DEMO = Path("src/voice_agent/server/static/demo.html")


def test_browser_playback_uses_the_worklet_queue() -> None:
    html = DEMO.read_text(encoding="utf-8")

    assert "this.play = [];" in html
    assert "this.port.onmessage" in html
    assert "workletNode.port.postMessage({ playback: pcm.buffer }" in html
    assert "createBufferSource" not in html


def test_worklet_captures_and_outputs_audio() -> None:
    html = DEMO.read_text(encoding="utf-8")

    assert "process(inputs, outputs)" in html
    assert "const channels = outputs[0] || [];" in html
    assert "for (const channel of channels)" in html
    assert "const input = inputs[0][0];" in html
    assert "workletNode.connect(ctx.destination);" in html


def test_worklet_resamples_browser_audio_to_server_rate() -> None:
    html = DEMO.read_text(encoding="utf-8")

    assert "this.captureRatio = sampleRate / ${SAMPLE_RATE};" in html
    assert "this.playRatio = ${SAMPLE_RATE} / sampleRate;" in html
    assert "this.capturePos += this.captureRatio;" in html
    assert "this.playPos += this.playRatio;" in html


def test_demo_page_shows_voice_prompt_suggestions() -> None:
    html = DEMO.read_text(encoding="utf-8")

    assert "Try asking" in html
    assert "Give me a summary of all my accounts." in html
    assert "Search my notes for unhappy customers." in html


def test_demo_page_animates_the_live_pipeline() -> None:
    html = DEMO.read_text(encoding="utf-8")

    assert 'class="flow"' in html
    assert 'data-flow="listening"' in html
    assert 'data-flow="asr"' in html
    assert 'data-flow="tool"' in html
    assert 'data-flow="tts"' in html
    assert "function setPhase(phase)" in html
    assert "setPhase('tts')" in html


def test_demo_page_switches_dashboard_copy_for_german_asr() -> None:
    html = DEMO.read_text(encoding="utf-8")

    assert "const COPY = {" in html
    assert "de: {" in html
    assert "Beispielfragen" in html
    assert "Want me to speak English? Just start speaking English." in html
    assert "function setLanguage(language)" in html
    assert "setLanguage(event.language);" in html
    assert 'data-i18n="try_asking"' in html


def test_demo_page_signposts_bilingual_mode() -> None:
    html = DEMO.read_text(encoding="utf-8")

    assert 'class="language-hint"' in html
    assert "Soll ich Deutsch sprechen? Sprich einfach Deutsch." in html
    assert 'data-i18n="language_hint"' in html


def test_demo_page_animates_langgraph_tool_calls() -> None:
    html = DEMO.read_text(encoding="utf-8")

    assert 'id="tool-popover"' in html
    assert "LangGraph tool call" in html
    assert "function showToolPopover(name, args)" in html
    assert "showToolPopover(event.name, event.arguments);" in html
    assert "hideToolPopover();" in html


def test_demo_page_links_a_favicon() -> None:
    html = DEMO.read_text(encoding="utf-8")

    assert '<link rel="icon" href="/favicon.svg" type="image/svg+xml" />' in html
