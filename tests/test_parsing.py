import pytest

from retro_eval.harness.parsing import (
    extract_json_object,
    parse_chemllm_answer,
    parse_cot_answer,
    parse_json_reactants_response,
    parse_rerank_answer,
)

TARGET_SMILES = "CC(=O)OCC"


def test_extract_json_object_handles_markdown_fence():
    raw = '```json\n{"reactants": ["CCO"], "reaction_class": "esterification"}\n```'
    assert extract_json_object(raw) == {
        "reactants": ["CCO"],
        "reaction_class": "esterification",
    }


def test_extract_json_object_handles_plain_json():
    raw = '{"reagents": "NaOH", "solvent": "water"}'
    assert extract_json_object(raw) == {"reagents": "NaOH", "solvent": "water"}


def test_extract_json_object_raises_without_braces():
    with pytest.raises(ValueError):
        extract_json_object("not json at all")


def test_parse_cot_answer_extracts_think_and_valid_precursors():
    raw = "<think>SN2 disconnection</think><answer>CCO.CC(=O)Cl</answer>"
    predicted, extra = parse_cot_answer(raw, TARGET_SMILES)
    assert predicted == "CCO.CC(=O)Cl"
    assert extra["think"] == "SN2 disconnection"
    assert extra["errors"] == []


def test_parse_cot_answer_reports_errors_for_invalid_fragment():
    raw = "<think>bad</think><answer>not-a-smiles</answer>"
    predicted, extra = parse_cot_answer(raw, TARGET_SMILES)
    assert predicted == "not-a-smiles"
    assert extra["errors"]


def test_parse_chemllm_answer_extracts_only_final_answer_line():
    raw = "Think: Disconnect the ester.\nAnswer: CCO.CC(=O)Cl"
    predicted, extra = parse_chemllm_answer(raw, TARGET_SMILES)
    assert predicted == "CCO.CC(=O)Cl"
    assert extra["candidate_answer"] == "CCO.CC(=O)Cl"
    assert extra["errors"] == []


def test_parse_chemllm_answer_does_not_treat_prose_as_smiles():
    predicted, extra = parse_chemllm_answer("Think: Disconnect the ester.", TARGET_SMILES)
    assert predicted == ""
    assert "did not contain" in extra["errors"][0]


def test_parse_json_reactants_response_canonicalizes_and_reports_reaction_class():
    raw = '{"reactants": ["CCO", "CC(=O)Cl"], "reaction_class": "esterification"}'
    predicted, extra = parse_json_reactants_response(raw)
    assert predicted == "CCO.CC(=O)Cl"
    assert extra["reaction_class"] == "esterification"
    assert extra["warnings"] == []


def test_parse_json_reactants_response_drops_unparseable_fragments():
    raw = '{"reactants": ["CCO", "not-a-smiles((("]}'
    predicted, extra = parse_json_reactants_response(raw)
    assert predicted == "CCO"
    assert "Dropped 1" in extra["warnings"][0]


RERANK_CANDIDATES = ["CCO", "CC(=O)Cl", "c1ccccc1"]


def test_parse_rerank_answer_matches_exact_candidate():
    raw = "<think>The alcohol plus acid chloride gives the ester.</think><answer>CC(=O)Cl</answer>"
    predicted, extra = parse_rerank_answer(raw, RERANK_CANDIDATES)
    assert predicted == "CC(=O)Cl"
    assert extra["think"] == "The alcohol plus acid chloride gives the ester."
    assert extra["warnings"] == []


def test_parse_rerank_answer_matches_canonically_equivalent_candidate():
    raw = "<answer>OCC</answer>"  # canonically equal to candidate "CCO"
    predicted, extra = parse_rerank_answer(raw, RERANK_CANDIDATES)
    assert predicted == "CCO"
    assert extra["warnings"] == []


def test_parse_rerank_answer_falls_back_to_candidate_1_without_answer_tag():
    predicted, extra = parse_rerank_answer("I think it's the ester.", RERANK_CANDIDATES)
    assert predicted == RERANK_CANDIDATES[0]
    assert "did not contain an <answer> tag" in extra["warnings"][0]


def test_parse_rerank_answer_falls_back_to_candidate_1_for_invalid_smiles():
    raw = "<answer>not-a-smiles(((</answer>"
    predicted, extra = parse_rerank_answer(raw, RERANK_CANDIDATES)
    assert predicted == RERANK_CANDIDATES[0]
    assert "failed RDKit validation" in extra["warnings"][0]


def test_parse_rerank_answer_uses_literal_answer_when_not_a_shown_candidate():
    raw = "<answer>CCN</answer>"  # valid SMILES, but not one of the shown candidates
    predicted, extra = parse_rerank_answer(raw, RERANK_CANDIDATES)
    assert predicted == "CCN"
    assert "did not exactly match" in extra["warnings"][0]


def test_parse_rerank_answer_skips_invalid_candidates_when_falling_back():
    candidates = ["not-a-smiles(((", "CCO", "c1ccccc1"]
    predicted, extra = parse_rerank_answer("no answer tag here", candidates)
    assert predicted == "CCO"  # first *valid* candidate, not candidates[0]
    assert "Fell back to the first valid T5 candidate" in extra["warnings"][0]


def test_parse_rerank_answer_reports_failure_when_all_candidates_invalid():
    candidates = ["not-a-smiles(((", "also-bad((", "still-bad(("]
    predicted, extra = parse_rerank_answer("no answer tag here", candidates)
    assert predicted == ""
    assert extra["errors"]
