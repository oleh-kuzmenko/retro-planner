"""Retry wrapper for network-backed `LLMProvider`s (Ollama connection errors, Groq rate limits).

`tenacity` is imported lazily inside `generate()`, not at module scope, so
importing this module (and anything that imports it) never requires
`tenacity` to be installed unless a retrying call is actually made.

Free-tier Groq 429s can demand waits far longer than a normal exponential
backoff (minutes for per-minute limits, hours for daily token/request
limits) -- long enough that blocking a script's process isn't always the
right call. `RetryingProvider` handles the short end of that itself (sleep
and retry transparently); once the required wait crosses
`rate_limit_auto_wait_seconds` it raises `ProviderPaused` instead, so a
caller that can persist progress (`scripts/models/run_rag_cot_groq.py`) can
save state and let the user resume later rather than blocking indefinitely.
"""

from __future__ import annotations

import logging
import re
import time

from retro_eval.providers import LLMProvider

LOGGER = logging.getLogger(__name__)

RATE_LIMIT_STATUS_CODE = 429

_RETRY_AFTER_PATTERN = re.compile(
    r"try again in\s+(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
    re.IGNORECASE,
)


class ProviderPaused(Exception):
    """Raised instead of the underlying error when a provider needs a long pause.

    `wait_seconds` is the provider's own estimate (parsed from a `Retry-After`
    header or an error message) of how long to wait before retrying, so the
    caller can log/act on it instead of getting a bare rate-limit exception.
    """

    def __init__(self, message: str, wait_seconds: float):
        super().__init__(message)
        self.wait_seconds = wait_seconds


def _is_rate_limit_error(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == RATE_LIMIT_STATUS_CODE


def _parse_retry_after_seconds(exc: BaseException) -> float | None:
    """Best-effort extraction of a suggested wait time from a 429 response.

    Tries the standard `Retry-After` header first, then falls back to
    parsing Groq/OpenAI-style messages like "Please try again in 2m59.56s".
    Returns `None` if neither is present/parseable.
    """
    response = getattr(exc, "response", None)
    header = None
    if response is not None:
        header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass

    match = _RETRY_AFTER_PATTERN.search(str(exc))
    if match and any(match.groups()):
        hours, minutes, seconds = match.group("hours"), match.group("minutes"), match.group("seconds")
        return float(hours or 0) * 3600 + float(minutes or 0) * 60 + float(seconds or 0)
    return None


class RetryingProvider:
    """Wraps any `LLMProvider` with exponential-backoff retry on connection errors/rate limits.

    Rate limits (HTTP 429) skip the generic exponential backoff entirely --
    it's pointless to retry a rate limit after a few seconds -- and instead
    either sleep for the provider-suggested wait (if short enough) or raise
    `ProviderPaused` for the caller to handle.
    """

    def __init__(
        self,
        inner: LLMProvider,
        max_attempts: int = 5,
        wait_multiplier: float = 2,
        wait_min: float = 2,
        wait_max: float = 60,
        rate_limit_auto_wait_seconds: float = 300,
        rate_limit_default_wait_seconds: float = 3600,
    ):
        self._inner = inner
        self._max_attempts = max_attempts
        self._wait_multiplier = wait_multiplier
        self._wait_min = wait_min
        self._wait_max = wait_max
        self._rate_limit_auto_wait_seconds = rate_limit_auto_wait_seconds
        self._rate_limit_default_wait_seconds = rate_limit_default_wait_seconds

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
            retry=tenacity.retry_if_exception(lambda exc: not _is_rate_limit_error(exc)),
            before_sleep=tenacity.before_sleep_log(LOGGER, logging.WARNING),
            reraise=True,
        )
        def _call() -> str:
            return self._inner.generate(
                messages=messages, model=model, temperature=temperature, json_mode=json_mode
            )

        while True:
            try:
                return _call()
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise
                wait_seconds = _parse_retry_after_seconds(exc) or self._rate_limit_default_wait_seconds
                if wait_seconds <= self._rate_limit_auto_wait_seconds:
                    LOGGER.warning(
                        "Rate limited (429); sleeping %.0fs before retrying automatically.",
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue
                raise ProviderPaused(str(exc), wait_seconds=wait_seconds) from exc
