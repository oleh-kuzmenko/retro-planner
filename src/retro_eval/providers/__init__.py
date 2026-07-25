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
    ) -> str:
        """Return raw model text for the supplied chat messages."""
        ...
