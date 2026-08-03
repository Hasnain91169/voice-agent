"""Handshake tokens for the media WebSocket.

Without this, anyone who finds the host can open a socket and occupy a GPU
indefinitely. The gateway this replaces left every endpoint unauthenticated and
bound to ``0.0.0.0``.

A short-lived signed token rather than a session cookie or a bearer header,
because the browser WebSocket API cannot set request headers — the token has to
survive in the URL, so it must be worthless within minutes of being issued.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import time

log = logging.getLogger(__name__)

_SEPARATOR = "."


def issue_token(secret: str, *, ttl_s: int, subject: str = "session") -> str:
    """Mint a token valid for ``ttl_s`` seconds."""
    expires = int(time.time()) + ttl_s
    nonce = secrets.token_urlsafe(8)
    payload = f"{subject}{_SEPARATOR}{expires}{_SEPARATOR}{nonce}"
    return f"{payload}{_SEPARATOR}{_sign(secret, payload)}"


def verify_token(secret: str, token: str, *, subject: str = "session") -> bool:
    """Check signature, subject and expiry."""
    try:
        raw_subject, raw_expiry, nonce, signature = token.split(_SEPARATOR)
    except ValueError:
        return False

    payload = f"{raw_subject}{_SEPARATOR}{raw_expiry}{_SEPARATOR}{nonce}"
    # Constant-time comparison: a timing-variable check on a signature leaks it
    # one byte at a time.
    if not hmac.compare_digest(_sign(secret, payload), signature):
        return False
    if raw_subject != subject:
        return False
    try:
        expires = int(raw_expiry)
    except ValueError:
        return False
    return time.time() < expires


def _sign(secret: str, payload: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def ephemeral_secret() -> str:
    """A per-process secret, for local development only.

    Sessions issued under it stop working when the process restarts, which is
    correct: an unconfigured deployment should not be quietly accepting tokens
    that outlive it.
    """
    return secrets.token_urlsafe(32)
