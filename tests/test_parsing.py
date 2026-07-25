import pytest

from retro_eval.harness.parsing import (
    extract_json_object,
    parse_chemllm_answer,
    parse_cot_answer,
    parse_json_reactants_response,
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
