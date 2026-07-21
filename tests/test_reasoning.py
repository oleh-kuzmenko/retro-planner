from retro_planner.reasoning import parse_reasoning_response, validate_precursors


def test_parse_reasoning_response_extracts_think_and_answer():
    text = (
        "<think>The ester can be disconnected into the acid and the alcohol.</think>"
        "<answer>CC(=O)O.CCO</answer>"
    )
    result = parse_reasoning_response(text)

    assert result.think == "The ester can be disconnected into the acid and the alcohol."
    assert result.answer_smiles == ["CC(=O)O", "CCO"]


def test_parse_reasoning_response_accepts_reason_tag():
    text = "<reason>Some analysis.</reason><answer>CCO</answer>"
    result = parse_reasoning_response(text)

    assert result.think == "Some analysis."
    assert result.answer_smiles == ["CCO"]


def test_parse_reasoning_response_falls_back_when_no_answer_tag():
    text = "CC(=O)O.CCO"
    result = parse_reasoning_response(text)

    assert result.think is None
    assert result.answer_smiles == ["CC(=O)O", "CCO"]


def test_parse_reasoning_response_strips_reaction_arrow_from_answer():
    text = "<answer>CC(=O)O.CCO>>CC(=O)OCC</answer>"
    result = parse_reasoning_response(text)

    assert result.answer_smiles == ["CC(=O)O", "CCO"]


def test_validate_precursors_accepts_valid_smiles():
    precursors, warnings, errors = validate_precursors(
        ["CC(=O)O", "CCO"],
        target_smiles="CC(=O)OCC",
    )

    assert precursors == ["CC(=O)O", "CCO"]
    assert errors == []


def test_validate_precursors_rejects_invalid_smiles():
    precursors, warnings, errors = validate_precursors(
        ["not-a-smiles"],
        target_smiles="CC(=O)OCC",
    )

    assert precursors is None
    assert errors


def test_validate_precursors_rejects_empty_answer():
    precursors, warnings, errors = validate_precursors([], target_smiles="CCO")

    assert precursors is None
    assert errors


def test_validate_precursors_warns_on_mass_imbalance():
    precursors, warnings, errors = validate_precursors(
        ["C"],
        target_smiles="c1ccc2c(c1)ccc1c2ccc2c1cccc2",
    )

    assert precursors == ["C"]
    assert errors == []
    assert any("mass" in warning for warning in warnings)
