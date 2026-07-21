"""LLM provider registry: chat-API backends plus local/research model backends.

Every provider implements the same `LLMProvider.generate()` contract regardless
of what runs underneath (a hosted chat API, a local HF seq2seq model, a LoRA
adapter pair, or a GGUF checkpoint via llama.cpp). Adding a new model is one
class with a `generate()` method plus one `LLM_PROVIDER_REGISTRY` entry; no
changes are needed in planning.py, retrieval.py, or streamlit_views.py.
"""

import re
from dataclasses import dataclass
from typing import Callable, Protocol


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


CATEGORY_CLOUD_API = "cloud_api"
CATEGORY_LOCAL_RESEARCH = "local_research"

CATEGORY_LABELS = {
    CATEGORY_CLOUD_API: "Cloud API",
    CATEGORY_LOCAL_RESEARCH: "Local / research",
}


@dataclass(frozen=True)
class LLMProviderConfig:
    key: str
    label: str
    api_key_env_var: str
    model_env_var: str
    default_model: str
    create_provider: Callable[[str, str | None], LLMProvider]
    key_url: str | None = None
    api_key_required: bool = True
    base_url_env_var: str | None = None
    default_base_url: str | None = None
    category: str = CATEGORY_CLOUD_API


_TARGET_SMILES_RE = re.compile(r"Target molecule \(SMILES\):\s*(\S+)")


def extract_target_smiles(prompt: str) -> str:
    """Pull the target SMILES back out of a build_cot_prompt()/-repair-prompt() string.

    Local providers fine-tuned on narrow JSON micro-prompts (ReactionT5, the
    two-stage Qwen LoRA) can't consume the general 4-block CoT prompt text
    directly; they only need the target molecule out of its `[Input]` block.
    """
    match = _TARGET_SMILES_RE.search(prompt)
    if match is None:
        raise ValueError(
            "Could not find 'Target molecule (SMILES): ...' in the prompt; "
            "expected output of build_cot_prompt()/build_cot_repair_prompt()."
        )
    return match.group(1)


from retro_planner.providers.chat_api import CHAT_API_PROVIDERS
from retro_planner.providers.local_causal import LOCAL_CAUSAL_PROVIDERS
from retro_planner.providers.local_gguf import LOCAL_GGUF_PROVIDERS
from retro_planner.providers.local_seq2seq import LOCAL_SEQ2SEQ_PROVIDERS

LLM_PROVIDER_REGISTRY: dict[str, LLMProviderConfig] = {
    **CHAT_API_PROVIDERS,
    **LOCAL_SEQ2SEQ_PROVIDERS,
    **LOCAL_CAUSAL_PROVIDERS,
    **LOCAL_GGUF_PROVIDERS,
}
