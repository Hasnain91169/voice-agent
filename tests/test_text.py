"""Tests for clause assembly and speech sanitising."""

from __future__ import annotations

import numpy as np

from voice_agent.audio import wav
from voice_agent.config import SAMPLE_RATE
from voice_agent.providers.tts_piper import _EdgeTrimmer
from voice_agent.text import ClauseAssembler, clean_for_speech


class TestCleanForSpeech:
    def test_strips_markdown_emphasis_but_keeps_the_words(self) -> None:
        assert clean_for_speech("That is **very** important") == "That is very important"

    def test_removes_bullets_and_headings(self) -> None:
        cleaned = clean_for_speech("## Options\n- first\n- second")
        assert "#" not in cleaned and "-" not in cleaned
        assert "first" in cleaned and "second" in cleaned

    def test_drops_code_fences_entirely(self) -> None:
        # Reading a code block aloud is never the right answer.
        cleaned = clean_for_speech("Try this:\n```python\nprint('hi')\n```\nDone.")
        assert "print" not in cleaned
        assert "Done." in cleaned

    def test_unwraps_inline_code(self) -> None:
        assert clean_for_speech("Run `status` now") == "Run status now"

    def test_removes_a_leading_role_label(self) -> None:
        assert clean_for_speech("Assistant: hello there") == "hello there"

    def test_replaces_characters_that_are_read_aloud(self) -> None:
        cleaned = clean_for_speech("wait—no… ok")
        assert "—" not in cleaned
        assert "…" not in cleaned

    def test_collapses_repeated_terminal_punctuation(self) -> None:
        assert clean_for_speech("Really!!!") == "Really!"

    def test_caps_length(self) -> None:
        assert len(clean_for_speech("word " * 500)) <= 600

    def test_empty_input(self) -> None:
        assert clean_for_speech("") == ""
        assert clean_for_speech("   ") == ""


class TestClauseAssembler:
    def test_emits_on_a_sentence_boundary(self) -> None:
        assembler = ClauseAssembler(min_chars=5)
        assert assembler.push("Hello there. ") == ["Hello there."]

    def test_holds_text_that_is_too_short_to_speak(self) -> None:
        # "Hi." on its own synthesises badly and sounds clipped.
        assembler = ClauseAssembler(min_chars=20)
        assert assembler.push("Hi. ") == []

    def test_cuts_at_a_comma_so_first_audio_is_not_gated_on_the_full_sentence(
        self,
    ) -> None:
        # The whole reason clause-level chunking exists: this sentence takes
        # seconds to generate, and the caller should not wait for the full stop.
        assembler = ClauseAssembler(min_chars=12)
        emitted = assembler.push(
            "I can see three open orders on your account, and the most recent "
        )
        assert emitted == ["I can see three open orders on your account,"]

    def test_accumulates_across_token_sized_pushes(self) -> None:
        assembler = ClauseAssembler(min_chars=10)
        emitted: list[str] = []
        for token in ["Your ", "order ", "shipped ", "on ", "Tuesday", ". "]:
            emitted += assembler.push(token)
        assert emitted == ["Your order shipped on Tuesday."]

    def test_time_flush_prevents_silence_on_unpunctuated_output(self) -> None:
        # A model that runs on without punctuation must not leave the caller
        # waiting for a comma that never arrives.
        assembler = ClauseAssembler(min_chars=10, flush_after_ms=400)
        assert assembler.push("this keeps going and going", now=100.0) == []
        emitted = assembler.push(" and going", now=100.5)
        assert emitted and "keeps going" in emitted[0]

    def test_time_flush_respects_the_minimum_length(self) -> None:
        assembler = ClauseAssembler(min_chars=30, flush_after_ms=100)
        assert assembler.push("too short", now=100.0) == []
        assert assembler.push("", now=200.0) == []

    def test_flush_returns_the_tail_regardless_of_length(self) -> None:
        # The stream has ended, so a short remainder is the rest of the answer,
        # not a premature cut.
        assembler = ClauseAssembler(min_chars=50)
        assembler.push("Yes")
        assert assembler.flush() == "Yes"

    def test_flush_is_empty_when_nothing_is_buffered(self) -> None:
        assert ClauseAssembler().flush() is None

    def test_multiple_clauses_from_one_push(self) -> None:
        assembler = ClauseAssembler(min_chars=5)
        assert assembler.push("First one. Second one. ") == [
            "First one.",
            "Second one.",
        ]

    def test_ends_at_boundary_detects_salvageable_text(self) -> None:
        # Used when a stream dies mid-response: text ending at a boundary can
        # be spoken, a mid-word fragment cannot.
        assembler = ClauseAssembler(min_chars=100)
        assembler.push("I checked your account,")
        assert assembler.ends_at_boundary() is True

    def test_ends_at_boundary_rejects_a_mid_word_fragment(self) -> None:
        assembler = ClauseAssembler(min_chars=100)
        assembler.push("I checked your acco")
        assert assembler.ends_at_boundary() is False

    def test_ends_at_boundary_on_empty_buffer(self) -> None:
        assert ClauseAssembler().ends_at_boundary() is False


class TestEdgeTrimming:
    """Piper puts silence either side of every utterance it synthesises.

    That is right for a sentence and wrong for a clause. The pipeline splits on
    commas, so the model's own trailing pause lands in the middle of a spoken
    sentence — measured at 922ms of dead air added to a three-clause answer
    that runs 6.8 seconds when synthesised whole.
    """

    @staticmethod
    def _tone(ms: int, amplitude: int = 6000) -> bytes:
        count = SAMPLE_RATE * ms // 1000
        samples = np.empty(count, dtype="<i2")
        samples[0::2] = amplitude
        samples[1::2] = -amplitude
        return samples.tobytes()

    @staticmethod
    def _hush(ms: int) -> bytes:
        return b"\x00\x00" * (SAMPLE_RATE * ms // 1000)

    def _run(self, pcm: bytes) -> bytes:
        trimmer = _EdgeTrimmer()
        out = b"".join(b"".join(trimmer.feed(pcm[i : i + 3200])) for i in range(0, len(pcm), 3200))
        return out + b"".join(trimmer.finish())

    def test_leading_silence_goes_entirely(self) -> None:
        """It is pure latency: the caller waits and hears nothing."""
        result = self._run(self._hush(200) + self._tone(400))
        assert wav.rms(result[:640]) > 1000

    def test_trailing_silence_is_cut_to_a_fixed_tail(self) -> None:
        """Kept short rather than removed: clauses run together without it."""
        result = self._run(self._tone(400) + self._hush(600))
        spoken = len(result) / (SAMPLE_RATE * 2) * 1000
        assert 400 <= spoken <= 400 + 120

    def test_speech_survives_intact(self) -> None:
        result = self._run(self._tone(500))
        assert len(result) >= SAMPLE_RATE * 2 * 480 // 1000

    def test_a_silent_clause_yields_nothing(self) -> None:
        """Better silence than a burst of noise floor."""
        assert self._run(self._hush(300)) == b""

    def test_internal_pauses_are_left_alone(self) -> None:
        """Only the edges. A pause inside a clause is how the words were said."""
        result = self._run(self._tone(200) + self._hush(150) + self._tone(200))
        spoken = len(result) / (SAMPLE_RATE * 2) * 1000
        assert spoken >= 540
