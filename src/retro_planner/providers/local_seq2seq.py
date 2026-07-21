"""Local seq2seq provider: ReactionT5v2 (`sagawa/ReactionT5v2-retrosynthesis*`).

Ported from fine-tune/v2/res/08_reactiont5_conditions_agent_demo.ipynb.
ReactionT5 takes the bare product SMILES as input and returns reactant SMILES
with no reasoning trace, so this is the "no-think" mode reasoning.py already
supports: the returned text has no <answer> tag, so parse_reasoning_response()
treats the whole response as the answer.

Heavy deps (torch, transformers) are only imported inside generate() /
_load_model(), matching the existing lazy-import pattern for groq/openai in
providers/chat_api.py, so the base install doesn't need the `local-models`
extra.
"""

import functools
import logging

from retro_planner.providers import (
    CATEGORY_LOCAL_RESEARCH,
    LLMProviderConfig,
    extract_target_smiles,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "sagawa/ReactionT5v2-retrosynthesis-USPTO_50k"


@functools.lru_cache(maxsize=2)
def _load_model(model_id: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=torch.float32)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tokenizer, model


class ReactionT5Provider:
    """Product SMILES in, reactant SMILES out; no <think> reasoning trace."""

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        import torch

        target_smiles = extract_target_smiles(messages[-1]["content"])
        tokenizer, seq2seq_model = _load_model(model or DEFAULT_MODEL)

        LOGGER.info("ReactionT5 request started model=%s target=%s", model, target_smiles)
        inputs = tokenizer([target_smiles], return_tensors="pt").to(seq2seq_model.device)
        with torch.no_grad():
            output_ids = seq2seq_model.generate(
                **inputs,
                num_beams=5,
                max_length=150,
                min_length=1,
            )
        # ReactionT5's tokenizer emits space-separated tokens; join before returning.
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        answer = decoded.strip().replace(" ", "")
        LOGGER.info("ReactionT5 response received model=%s answer=%s", model, answer)
        return answer


LOCAL_SEQ2SEQ_PROVIDERS: dict[str, LLMProviderConfig] = {
    "local_reactiont5": LLMProviderConfig(
        key="local_reactiont5",
        label="ReactionT5v2 (local, no reasoning trace)",
        api_key_env_var="HF_TOKEN",
        model_env_var="LOCAL_REACTIONT5_MODEL",
        default_model=DEFAULT_MODEL,
        create_provider=lambda _api_key, _base_url: ReactionT5Provider(),
        api_key_required=False,
        category=CATEGORY_LOCAL_RESEARCH,
    ),
}
