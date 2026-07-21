"""Local GGUF provider: ChemLLM-20B-Chat via llama-cpp-python.

Ported from base_models_res/ChemLLM-20B-Chat.ipynb. Unlike the ReactionT5 and
two-stage LoRA providers, llama.cpp's create_chat_completion() already speaks
the same messages-in/text-out shape as the hosted chat-API providers, so the
general 4-block CoT prompt from prompting.build_cot_prompt() is sent through
unchanged and the model's own <think>/<answer> tagged reply is parsed by the
same reasoning.py contract as every other provider.

Heavy deps (llama_cpp) are only imported inside generate() / _load_llama(),
matching the existing lazy-import pattern for groq/openai.
"""

import functools
import logging

from retro_planner.providers import CATEGORY_LOCAL_RESEARCH, LLMProviderConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_REPO_ID = "mradermacher/ChemLLM-20B-Chat-SFT-i1-GGUF"
DEFAULT_FILENAME = "ChemLLM-20B-Chat-SFT.i1-Q4_K_M.gguf"


@functools.lru_cache(maxsize=1)
def _load_llama(repo_id: str, filename: str):
    from llama_cpp import Llama

    return Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_ctx=4096,
        n_gpu_layers=25,
        verbose=False,
    )


class ChemLLMGGUFProvider:
    """Local GGUF chat provider for ChemLLM-20B-Chat via llama-cpp-python."""

    def __init__(self, repo_id: str = DEFAULT_REPO_ID, filename: str = DEFAULT_FILENAME):
        self.repo_id = repo_id
        self.filename = filename

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        llama = _load_llama(self.repo_id, self.filename)
        LOGGER.info(
            "ChemLLM GGUF request started repo_id=%s temperature=%.2f messages=%d",
            self.repo_id,
            temperature,
            len(messages),
        )
        completion = llama.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=1500,
        )
        content = completion["choices"][0]["message"]["content"].strip()
        LOGGER.info("ChemLLM GGUF response received repo_id=%s response_chars=%d", self.repo_id, len(content))
        return content


LOCAL_GGUF_PROVIDERS: dict[str, LLMProviderConfig] = {
    "local_chemllm_gguf": LLMProviderConfig(
        key="local_chemllm_gguf",
        label="ChemLLM-20B-Chat GGUF (local)",
        api_key_env_var="HF_TOKEN",
        model_env_var="LOCAL_CHEMLLM_MODEL",
        default_model=DEFAULT_REPO_ID,
        create_provider=lambda _api_key, _base_url: ChemLLMGGUFProvider(),
        api_key_required=False,
        category=CATEGORY_LOCAL_RESEARCH,
    ),
}
