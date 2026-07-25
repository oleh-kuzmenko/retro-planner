"""Retry wrapper for network-backed `LLMProvider`s (Ollama connection errors, Groq rate limits).

`tenacity` is imported lazily inside `generate()`, not at module scope, so
importing this module (and anything that imports it) never requires
`tenacity` to be installed unless a retrying call is actually made.
"""

from __future__ import annotations

import logging

from retro_eval.providers import LLMProvider

LOGGER = logging.getLogger(__name__)


class RetryingProvider:
    """Wraps any `LLMProvider` with exponential-backoff retry on connection errors/rate limits."""

    def __init__(
        self,
        inner: LLMProvider,
        max_attempts: int = 5,
        wait_multiplier: float = 2,
        wait_min: float = 2,
        wait_max: float = 60,
    ):
        self._inner = inner
        self._max_attempts = max_attempts
        self._wait_multiplier = wait_multiplier
        self._wait_min = wait_min
        self._wait_max = wait_max

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        import tenacity

        @tenacity.retry(
            wait=tenacity.wait_exponential(
                multiplier=self._wait_multiplier, min=self._wait_min, max=self._wait_max
            ),
            stop=tenacity.stop_after_attempt(self._max_attempts),
            before_sleep=tenacity.before_sleep_log(LOGGER, logging.WARNING),
            reraise=True,
        )
        def _call() -> str:
            return self._inner.generate(
                messages=messages, model=model, temperature=temperature, json_mode=json_mode
            )

        return _call()
