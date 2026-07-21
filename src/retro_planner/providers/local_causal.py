"""Local two-stage fine-tuned Qwen2.5-7B LoRA provider (reactants/class + conditions).

Ported from fine-tune/v2/res/06_agent_inference_demo.ipynb. Both LoRA adapters
were trained on narrow JSON micro-prompts, not the general 4-block CoT
template used elsewhere in the app, so this provider extracts the target
SMILES out of the incoming prompt, runs the fixed two-stage pipeline
internally, and formats the combined result as <think>/<answer> text so the
same reasoning.py parser/validator handles it like every other provider.

Heavy deps (torch, transformers, peft) are only imported inside generate() /
_load_adapter(), matching the existing lazy-import pattern for groq/openai.
"""

import functools
import json
import logging
import re

from retro_planner.providers import (
    CATEGORY_LOCAL_RESEARCH,
    LLMProviderConfig,
    extract_target_smiles,
)

LOGGER = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
REACTANT_LORA_ID = "oleh13/retro-reactants-qwen25-7b-lora"
CONDITION_LORA_ID = "oleh13/retro-conditions-qwen25-7b-lora"

REACTANT_SYSTEM = """You are a chemistry reaction prediction model.
Return valid compact JSON only.
Predict reactants and reaction class for a single-step synthesis of the target product.
Use canonical SMILES strings when possible.
Do not include conditions or explanations."""

CONDITION_SYSTEM = """You are a chemistry condition recommendation model.
Return valid compact JSON only.
Given product SMILES, reactants, and reaction class, predict practical reaction conditions.
Do not invent evidence IDs and do not include explanations."""


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError(f"No JSON object found in LoRA output: {cleaned[:300]}")
    return json.loads(cleaned[first : last + 1])


@functools.lru_cache(maxsize=2)
def _load_adapter(adapter_id: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_id)
    model.eval()
    return tokenizer, model


def _generate_json(tokenizer, model, messages: list[dict[str, str]], max_new_tokens: int) -> dict:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    raw = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()
    return _extract_json(raw)


def _canonicalize_reactants(reactants: list, target_smiles: str) -> list[str]:
    from retro_planner.chemistry import canonicalize_smiles

    canonical = [canonicalize_smiles(reactant) for reactant in reactants]
    canonical = [smiles for smiles in canonical if smiles]
    if not canonical:
        raise ValueError(
            f"Two-stage LoRA predicted no valid reactants for target '{target_smiles}'."
        )
    return canonical


class TwoStageQwenLoraProvider:
    """Two-stage local LoRA pipeline: predict reactants/class, then conditions."""

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        target_smiles = extract_target_smiles(messages[-1]["content"])

        reactant_tokenizer, reactant_model = _load_adapter(REACTANT_LORA_ID)
        reactant_data = _generate_json(
            reactant_tokenizer,
            reactant_model,
            [
                {"role": "system", "content": REACTANT_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "predict_reactants_and_class",
                            "product_smiles": target_smiles,
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
            max_new_tokens=256,
        )
        reactants = _canonicalize_reactants(
            reactant_data.get("reactants") or [], target_smiles
        )
        reaction_class = str(reactant_data.get("reaction_class") or "unclassified")

        condition_tokenizer, condition_model = _load_adapter(CONDITION_LORA_ID)
        condition_data = _generate_json(
            condition_tokenizer,
            condition_model,
            [
                {"role": "system", "content": CONDITION_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "predict_conditions",
                            "product_smiles": target_smiles,
                            "reactants": reactants,
                            "reaction_class": reaction_class,
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
            max_new_tokens=320,
        )

        think = (
            f"Reaction class (predicted by reactant/class adapter): {reaction_class}\n"
            "Conditions (predicted by condition adapter): "
            f"reagents={condition_data.get('reagents', 'not specified')}; "
            f"solvent={condition_data.get('solvent', 'not specified')}; "
            f"temperature={condition_data.get('temperature_celsius', 'not specified')}; "
            f"time={condition_data.get('reaction_time', 'not specified')}; "
            f"atmosphere={condition_data.get('atmosphere', 'not specified')}; "
            f"workup={condition_data.get('workup_purification', 'not specified')}; "
            f"yield={condition_data.get('expected_yield_percent', 'not specified')}"
        )
        answer = ".".join(reactants)
        return f"<think>{think}</think>\n<answer>{answer}</answer>"


LOCAL_CAUSAL_PROVIDERS: dict[str, LLMProviderConfig] = {
    "local_qwen_two_stage": LLMProviderConfig(
        key="local_qwen_two_stage",
        label="Qwen2.5-7B two-stage LoRA (local)",
        api_key_env_var="HF_TOKEN",
        model_env_var="LOCAL_QWEN_LORA_MODEL",
        default_model=BASE_MODEL,
        create_provider=lambda _api_key, _base_url: TwoStageQwenLoraProvider(),
        api_key_required=False,
        category=CATEGORY_LOCAL_RESEARCH,
    ),
}
