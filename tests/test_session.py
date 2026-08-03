"""Tests for per-call state, and for the record of what the caller heard."""

from __future__ import annotations

from voice_agent.agent.memory import ConversationStore
from voice_agent.config import BYTES_PER_FRAME
from voice_agent.session import SpokenTracker, TurnMetrics


def frames(count: int) -> int:
    """Byte length of ``count`` frames of audio."""
    return count * BYTES_PER_FRAME


class TestSpokenTracker:
    """What gets committed to history after an interruption.

    If the full generated text were committed, the agent would spend the rest
    of the call believing it had told the caller things they never heard.
    """

    def test_everything_played_returns_everything(self) -> None:
        tracker = SpokenTracker()
        tracker.add("Your order shipped.", frames(50))
        tracker.add("It arrives Tuesday.", frames(50))
        assert tracker.spoken(frames(100)) == "Your order shipped. It arrives Tuesday."

    def test_overrun_does_not_invent_text(self) -> None:
        tracker = SpokenTracker()
        tracker.add("All done.", frames(10))
        assert tracker.spoken(frames(999)) == "All done."

    def test_interruption_drops_clauses_never_played(self) -> None:
        tracker = SpokenTracker()
        tracker.add("Your order shipped.", frames(50))
        tracker.add("It arrives Tuesday.", frames(50))
        # Cut cleanly at the end of the first clause.
        assert tracker.spoken(frames(50)) == "Your order shipped."

    def test_partial_clause_is_truncated_at_a_word_boundary(self) -> None:
        tracker = SpokenTracker()
        tracker.add("The quick brown fox jumps over", frames(100))
        spoken = tracker.spoken(frames(50))
        assert spoken  # something was heard
        assert spoken in "The quick brown fox jumps over"
        # A half-word is neither honest nor readable.
        assert not spoken.endswith(" ")
        assert all(word in "The quick brown fox jumps over" for word in spoken.split())

    def test_nothing_played_returns_nothing(self) -> None:
        tracker = SpokenTracker()
        tracker.add("Never heard.", frames(50))
        assert tracker.spoken(0) == ""

    def test_zero_length_audio_is_ignored(self) -> None:
        tracker = SpokenTracker()
        tracker.add("silent", 0)
        assert tracker.clauses == []

    def test_total_bytes_sums_clauses(self) -> None:
        tracker = SpokenTracker()
        tracker.add("one", frames(10))
        tracker.add("two", frames(15))
        assert tracker.total_bytes == frames(25)


class TestTurnMetrics:
    def test_summary_reports_every_budget_stage(self) -> None:
        metrics = TurnMetrics(
            endpoint_ms=250,
            asr_ms=84,
            llm_first_token_ms=98,
            first_clause_ms=80,
            tts_first_chunk_ms=113,
            first_audio_ms=422,
        )
        summary = metrics.summary()
        for field in ("endpoint", "asr", "ttft", "clause", "tts", "first_audio"):
            assert field in summary

    def test_barge_in_and_events_are_surfaced(self) -> None:
        metrics = TurnMetrics()
        metrics.barged_in = True
        metrics.note("filler")
        summary = metrics.summary()
        assert "barged_in" in summary
        assert "filler" in summary


class TestConversationStore:
    def test_history_round_trips(self) -> None:
        store = ConversationStore()
        store.commit_turn("t1", "Hello", "Hi, how can I help?")
        history = store.history("t1")
        assert [m.role for m in history] == ["user", "assistant"]
        assert history[1].content == "Hi, how can I help?"

    def test_threads_are_isolated(self) -> None:
        # The bug this guards against is the one that made the previous system
        # unusable with two concurrent calls.
        store = ConversationStore()
        store.commit_turn("a", "I am Alice", "Hello Alice")
        store.commit_turn("b", "I am Bob", "Hello Bob")
        assert "Alice" in store.history("a")[0].content
        assert "Alice" not in " ".join(m.content for m in store.history("b"))

    def test_empty_sides_are_not_recorded(self) -> None:
        # A barged-in turn where nothing was heard must not add an empty
        # assistant message.
        store = ConversationStore()
        store.commit_turn("t1", "Hello", "")
        assert [m.role for m in store.history("t1")] == ["user"]

    def test_window_bounds_growth(self) -> None:
        store = ConversationStore(window_turns=3)
        for i in range(10):
            store.commit_turn("t1", f"q{i}", f"a{i}")
        history = store.history("t1")
        assert len(history) == 6  # 3 turns, two messages each
        assert history[-1].content == "a9"

    def test_facts_survive_the_message_window(self) -> None:
        store = ConversationStore(window_turns=1)
        store.remember("t1", "caller_name", "Alex")
        for i in range(5):
            store.commit_turn("t1", f"q{i}", f"a{i}")
        assert store.facts("t1")["caller_name"] == "Alex"
