import re

from retro_eval import prompting
from retro_eval.prompting import build_cot_prompt, build_cot_repair_prompt, build_rerank_prompt


CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

TARGET_SMILES = "CC(=O)OCC"
RAG_EXAMPLES = [
    {
        "reaction_id": "rxn-1",
        "reaction_smiles": "CC(=O)O.CCO>>CC(=O)OCC",
        "product_smiles": "CC(=O)OCC",
        "reactants_smiles": "CC(=O)O.CCO",
        "reaction_class": "esterification",
        "final_hybrid_score": 0.87,
    }
]


def test_build_cot_prompt_has_four_blocks_in_order():
    prompt = build_cot_prompt(TARGET_SMILES, RAG_EXAMPLES)

    for marker in ("[System]", "[Context]", "[Instruction]", "[Input]"):
        assert marker in prompt

    assert prompt.index("[System]") < prompt.index("[Context]")
    assert prompt.index("[Context]") < prompt.index("[Instruction]")
    assert prompt.index("[Instruction]") < prompt.index("[Input]")


def test_build_cot_prompt_mentions_think_and_answer_tags():
    prompt = build_cot_prompt(TARGET_SMILES, RAG_EXAMPLES)

    assert "<think>" in prompt
    assert "<answer>" in prompt
    assert TARGET_SMILES in prompt


def test_build_cot_prompt_handles_no_rag_examples():
    prompt = build_cot_prompt(TARGET_SMILES, [])

    assert "[Context]" not in prompt
    assert "[System]" in prompt
    assert "[Instruction]" in prompt


def test_build_cot_repair_prompt_includes_issues():
    prompt = build_cot_repair_prompt(
        TARGET_SMILES,
        RAG_EXAMPLES,
        previous_response="<answer>garbage</answer>",
        issues=["Reactant SMILES 'garbage' failed RDKit validation."],
    )

    assert "garbage" in prompt
    assert "failed RDKit validation" in prompt


def test_build_rerank_prompt_lists_all_candidates_and_tags():
    candidates = ["CC(=O)O.CCO", "CCOC(=O)C", "CCO"]
    prompt = build_rerank_prompt(TARGET_SMILES, RAG_EXAMPLES, candidates)

    for idx, candidate in enumerate(candidates, start=1):
        assert f"Candidate {idx}: {candidate}" in prompt
    assert "<think>" in prompt
    assert "<answer>" in prompt
    assert TARGET_SMILES in prompt
    assert RAG_EXAMPLES[0]["reaction_smiles"] in prompt


def test_build_rerank_prompt_handles_no_rag_examples():
    prompt = build_rerank_prompt(TARGET_SMILES, [], ["CCO"])

    assert "No literature precedents were retrieved." in prompt
    assert "Candidate 1: CCO" in prompt


def test_build_rerank_prompt_flags_top_candidate_and_requires_reasoning():
    candidates = ["CC(=O)O.CCO", "CCOC(=O)C"]
    prompt = build_rerank_prompt(TARGET_SMILES, RAG_EXAMPLES, candidates)

    assert "Candidate 1: CC(=O)O.CCO <- specialized model's top-ranked" in prompt
    assert "Candidate 2: CCOC(=O)C" in prompt
    assert "Candidate 2: CCOC(=O)C <-" not in prompt
    assert "You MUST reason step by step" in prompt
    assert "clear, specific evidence" in prompt


def test_no_prompt_template_strings_contain_cyrillic_characters():
    with open(prompting.__file__, encoding="utf-8") as handle:
        module_source = handle.read()

    assert not CYRILLIC_RE.search(module_source), (
        "prompting.py must only ever emit English text to the LLM"
    )
