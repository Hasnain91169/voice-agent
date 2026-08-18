"""Pre-synthesised audio for the recovery paths.

Every failure path speaks. If it had to synthesise at the moment it fired, the
caller would hear silence during exactly the situation silence is worst: the
agent has already failed to respond, and now it pauses again while the fallback
is generated.

So the fixed lines are synthesised once at startup and held as PCM. Playing one
costs a queue append. The set is small and bounded — seven short lines,
including three time-aware greetings — so this is a few hundred kilobytes, not
a cache with an eviction policy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

from voice_agent.agent import prompts
from voice_agent.agent.prompts import CACHED_LINES
from voice_agent.config import BYTES_PER_FRAME, FRAME_MS
from voice_agent.providers.base import TTS

log = logging.getLogger(__name__)


class PromptCache:
    """Named lines of speech, synthesised ahead of time."""

    def __init__(self) -> None:
        #: Keyed by (language, line name). A recovery path fires at the worst
        #: possible moment, so the audio for every language has to exist before
        #: the first call rather than being synthesised when it is needed.
        self._audio: dict[tuple[str, str], bytes] = {}

    async def build(self, tts: TTS, languages: Sequence[str] = ("en",)) -> None:
        """Synthesise every cached line, in every language. Failures are logged,
        never fatal.

        A missing cached line degrades a recovery path to silence; a startup
        that refuses to boot because a fallback phrase could not be synthesised
        degrades the whole service. The health check reports the shortfall.

        Each language is synthesised with its own voice, which means asking the
        TTS adapter to switch — the alternative is a German sentence read by an
        English voice, which is worse than the English sentence would have been.
        """
        for language in languages:
            switch = getattr(tts, "use_language", None)
            if switch is not None:
                switch(language)
            for name in CACHED_LINES:
                text = prompts.line(name, language)
                try:
                    chunks = [chunk async for chunk in tts.synthesize(text)]
                except Exception as exc:
                    log.warning("could not pre-synthesise %r in %s: %s", name, language, exc)
                    continue
                audio = b"".join(chunks)
                if audio:
                    self._audio[(language, name)] = audio
        if (switch := getattr(tts, "use_language", None)) is not None:
            switch(languages[0])

        log.info(
            "prompt cache ready: %d/%d lines across %s (%.1f KB)",
            len(self._audio),
            len(CACHED_LINES) * len(languages),
            ",".join(languages),
            sum(len(a) for a in self._audio.values()) / 1024,
        )

    def get(self, name: str, language: str = "en") -> bytes | None:
        """The line in this language, falling back to English.

        Falling back matters: a language whose synthesis failed at startup
        should still get *a* recovery line rather than silence, which is the
        one failure mode the whole cache exists to prevent.
        """
        return self._audio.get((language, name)) or self._audio.get(("en", name))

    def greeting(self, language: str = "en", now: datetime | None = None) -> bytes | None:
        """Return the cached greeting appropriate for the call's local time."""
        return self.get(prompts.greeting_name(now), language)

    def duration_ms(self, name: str, language: str = "en") -> int:
        audio = self.get(name, language)
        if not audio:
            return 0
        return len(audio) // BYTES_PER_FRAME * FRAME_MS

    @property
    def ready(self) -> bool:
        """Every line exists in at least one language.

        Readiness is per line rather than per language-and-line: a missing
        German clarifier falls back to the English one and the call survives,
        where a missing clarifier in every language is silence.
        """
        return not self.missing

    @property
    def missing(self) -> list[str]:
        have = {name for _, name in self._audio}
        return [name for name in CACHED_LINES if name not in have]
