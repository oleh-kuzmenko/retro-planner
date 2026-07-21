import pytest

from retro_planner.prompting import build_cot_prompt, build_cot_repair_prompt
from retro_planner.providers import (
    CATEGORY_CLOUD_API,
    CATEGORY_LOCAL_RESEARCH,
    LLM_PROVIDER_REGISTRY,
    extract_target_smiles,
)
from retro_planner.providers.local_causal import _extract_json

TARGET_SMILES = "CC(=O)OCC"


def test_registry_contains_chat_api_and_local_providers():
    assert set(LLM_PROVIDER_REGISTRY) == {
        "groq",
        "openai",
        "custom_openai",
        "local_reactiont5",
        "local_qwen_two_stage",
        "local_chemllm_gguf",
    }


def test_registry_categories_split_cloud_api_from_local_research():
    # The sidebar groups providers by this field (Cloud API vs Local / research);
    # every provider must land in exactly one of the two known categories.
    for key in ("groq", "openai", "custom_openai"):
        assert LLM_PROVIDER_REGISTRY[key].category == CATEGORY_CLOUD_API
    for key in ("local_reactiont5", "local_qwen_two_stage", "local_chemllm_gguf"):
        assert LLM_PROVIDER_REGISTRY[key].category == CATEGORY_LOCAL_RESEARCH


def test_local_providers_do_not_require_an_api_key():
    for key in ("local_reactiont5", "local_qwen_two_stage", "local_chemllm_gguf"):
        assert LLM_PROVIDER_REGISTRY[key].api_key_required is False


def test_local_providers_are_constructible_without_heavy_deps_installed():
    # create_provider() must stay cheap (no model loading) so instantiating a
    # provider doesn't require torch/transformers/peft/llama_cpp to be
    # installed; only calling .generate() should need the `local-models` extra.
    for key in ("local_reactiont5", "local_qwen_two_stage", "local_chemllm_gguf"):
        config = LLM_PROVIDER_REGISTRY[key]
        provider = config.create_provider("", None)
        assert hasattr(provider, "generate")


def test_extract_target_smiles_from_cot_prompt():
    prompt = build_cot_prompt(TARGET_SMILES, [])
    assert extract_target_smiles(prompt) == TARGET_SMILES


def test_extract_target_smiles_from_repair_prompt():
    prompt = build_cot_repair_prompt(
        TARGET_SMILES,
        [],
        previous_response="<answer>garbage</answer>",
        issues=["invalid SMILES"],
    )
    assert extract_target_smiles(prompt) == TARGET_SMILES


def test_extract_target_smiles_raises_when_marker_missing():
    with pytest.raises(ValueError):
        extract_target_smiles("no target marker here")


def test_local_causal_extract_json_handles_markdown_fence():
    raw = '```json\n{"reactants": ["CCO"], "reaction_class": "esterification"}\n```'
    assert _extract_json(raw) == {
        "reactants": ["CCO"],
        "reaction_class": "esterification",
    }


def test_local_causal_extract_json_handles_plain_json():
    raw = '{"reagents": "NaOH", "solvent": "water"}'
    assert _extract_json(raw) == {"reagents": "NaOH", "solvent": "water"}


def test_local_causal_extract_json_raises_without_braces():
    with pytest.raises(ValueError):
        _extract_json("not json at all")
