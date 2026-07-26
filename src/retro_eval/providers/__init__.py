"""`LLMProvider` contract shared by every chat-API-backed model runner script.

Any object with a matching `generate()` method works as a provider — no
registry lookup needed, `scripts/models/*.py` construct the provider class
they want directly (see `chat_api.py`, `retrying.py`).
"""

from typing import Protocol


class LLMProvider(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
    ) -> str:
        """Return raw model text for the supplied chat messages.

        `reasoning_effort` (e.g. "low"/"medium"/"high") is forwarded to the API request
        only when set, for reasoning-capable models (gpt-oss, o-series, ...) where it
        controls how much hidden deliberation happens before the visible answer. Leave
        it `None` for models that don't support the field -- most hosted chat models
        (e.g. Groq's llama-3.3) reject unknown request parameters.
        """
        ...
