from retro_planner.prompts import (
    build_no_rag_system_prompt,
    build_rag_prompt,
    build_rag_system_prompt,
)


def test_system_prompts_preserve_json_only_contract():
    no_rag_prompt = build_no_rag_system_prompt()
    rag_prompt = build_rag_system_prompt()

    assert "valid JSON only" in no_rag_prompt
    assert "valid JSON only" in rag_prompt
    assert "product_smiles" in no_rag_prompt
    assert "product_smiles" in rag_prompt


def test_rag_prompt_preserves_target_and_evidence_instructions():
    prompt = build_rag_prompt(
        target_smiles="CC(=O)O",
        reactions=[
            {
                "reaction_id": "r1",
                "molecule_similarity": 0.8,
                "reaction_similarity": 0.4,
                "reaction_class_similarity": 1.0,
                "final_hybrid_score": 0.72,
                "reaction_smiles": "CCO>>CC=O",
                "reactants_smiles": "CCO",
                "product_smiles": "CC=O",
                "reaction_class": "oxidation",
            }
        ],
        optimization_objective="BALANCED",
        route_count=3,
    )

    assert "CC(=O)O" in prompt
    assert "Generate exactly 3 distinct" in prompt
    assert "Retrieved product_smiles values are example products only" in prompt
    assert "r1" in prompt
