"""Tests for the WebSocket handshake token.

Without this gate, anyone who finds the host can open a media socket and occupy
a GPU indefinitely — which is exactly what the unauthenticated gateway this
replaces allowed.
"""

from __future__ import annotations

import time

from voice_agent.server.security import ephemeral_secret, issue_token, verify_token

SECRET = "test-secret-value"


class TestTokens:
    def test_a_freshly_issued_token_verifies(self) -> None:
        assert verify_token(SECRET, issue_token(SECRET, ttl_s=60)) is True

    def test_a_token_from_another_secret_is_rejected(self) -> None:
        assert verify_token("different", issue_token(SECRET, ttl_s=60)) is False

    def test_tampering_with_the_payload_is_rejected(self) -> None:
        token = issue_token(SECRET, ttl_s=60)
        subject, expiry, nonce, signature = token.split(".")
        forged = f"{subject}.{int(expiry) + 3600}.{nonce}.{signature}"
        assert verify_token(SECRET, forged) is False

    def test_an_expired_token_is_rejected(self) -> None:
        # The token rides in a URL, because the browser WebSocket API cannot
        # set headers — so it must be worthless within minutes.
        assert verify_token(SECRET, issue_token(SECRET, ttl_s=-1)) is False

    def test_expiry_is_enforced_against_the_clock(self) -> None:
        token = issue_token(SECRET, ttl_s=1)
        assert verify_token(SECRET, token) is True
        assert verify_token(SECRET, token, subject="session") is True
        assert int(token.split(".")[1]) <= int(time.time()) + 1

    def test_subject_must_match(self) -> None:
        token = issue_token(SECRET, ttl_s=60, subject="session")
        assert verify_token(SECRET, token, subject="admin") is False

    def test_malformed_tokens_are_rejected_without_raising(self) -> None:
        for junk in ("", "not-a-token", "a.b.c", "a.b.c.d.e", "session.abc.n.sig"):
            assert verify_token(SECRET, junk) is False

    def test_tokens_are_unique_per_issue(self) -> None:
        # A replayed URL from a shared screenshot should not be a valid second
        # session for as long as the first.
        assert issue_token(SECRET, ttl_s=60) != issue_token(SECRET, ttl_s=60)

    def test_ephemeral_secret_is_unguessable_and_unique(self) -> None:
        first, second = ephemeral_secret(), ephemeral_secret()
        assert first != second
        assert len(first) >= 32
