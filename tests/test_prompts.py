"""Tests for fixed spoken prompts and time-aware call openings."""

from __future__ import annotations

from datetime import datetime

import pytest

from voice_agent.agent.prompts import greeting, greeting_name, greeting_period


@pytest.mark.parametrize(
    ("hour", "period"),
    [
        (0, "evening"),
        (4, "evening"),
        (5, "morning"),
        (11, "morning"),
        (12, "afternoon"),
        (17, "afternoon"),
        (18, "evening"),
        (23, "evening"),
    ],
)
def test_greeting_period_follows_local_hour(hour: int, period: str) -> None:
    assert greeting_period(datetime(2026, 8, 13, hour)) == period
    assert greeting_name(datetime(2026, 8, 13, hour)) == f"greeting_{period}"


def test_english_greeting_changes_with_time_of_day() -> None:
    assert greeting("en", datetime(2026, 8, 13, 9)) == "Good morning. What do you need?"
    assert greeting("en", datetime(2026, 8, 13, 14)) == "Good afternoon. What do you need?"
    assert greeting("en", datetime(2026, 8, 13, 19)) == "Good evening. What do you need?"


def test_german_afternoon_uses_natural_greeting() -> None:
    assert greeting("de", datetime(2026, 8, 13, 14)) == "Guten Tag. Was brauchst du?"
