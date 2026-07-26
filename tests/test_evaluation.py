from retro_eval.evaluation import (
    canonical_precursor_set,
    is_core_exact_match,
    is_core_exact_match_smiles,
    is_exact_match,
    is_exact_match_smiles,
    is_valid_smiles,
    split_fragments,
    strip_salts_and_catalysts,
    structure_success_rate,
)


def test_canonical_precursor_set_canonicalizes_and_dedupes():
    result = canonical_precursor_set(["CCO", "OCC"])
    assert result == frozenset({"CCO"})


def test_canonical_precursor_set_returns_none_on_invalid_fragment():
    assert canonical_precursor_set(["CCO", "not-a-smiles"]) is None


def test_is_exact_match_ignores_order():
    assert is_exact_match(["CCO", "CC(=O)O"], ["CC(=O)O", "OCC"]) is True


def test_is_exact_match_rejects_different_precursors():
    assert is_exact_match(["CCO"], ["CC(=O)O"]) is False


def test_is_exact_match_rejects_unparseable_prediction():
    assert is_exact_match(["not-a-smiles"], ["CCO"]) is False


def test_structure_success_rate_counts_parseable_smiles():
    assert structure_success_rate(["CCO", "not-a-smiles", "CC(=O)O.CCO"]) == 2 / 3


def test_structure_success_rate_treats_empty_string_as_failure():
    assert structure_success_rate(["CCO", ""]) == 0.5


def test_structure_success_rate_empty_list_returns_zero():
    assert structure_success_rate([]) == 0.0


def test_split_fragments_splits_on_dot():
    assert split_fragments("CCO.CC(=O)O") == ["CCO", "CC(=O)O"]


def test_split_fragments_handles_none_and_empty():
    assert split_fragments(None) == []
    assert split_fragments("") == []


def test_is_valid_smiles_true_for_parseable_dot_joined_string():
    assert is_valid_smiles("CCO.CC(=O)O") is True


def test_is_valid_smiles_false_for_garbage_or_empty():
    assert is_valid_smiles("not-a-smiles(((") is False
    assert is_valid_smiles("") is False
    assert is_valid_smiles(None) is False


def test_is_exact_match_smiles_ignores_order_and_canonicalizes():
    assert is_exact_match_smiles("CCO.CC(=O)O", "CC(=O)O.OCC") is True


def test_is_exact_match_smiles_false_for_empty_prediction_or_reference():
    assert is_exact_match_smiles("", "CCO") is False
    assert is_exact_match_smiles("CCO", None) is False


def test_strip_salts_and_catalysts_drops_ions_and_carbon_free_molecules():
    assert strip_salts_and_catalysts(["CCO", "[Na+]", "[OH-]", "O", "Cl"]) == ["CCO"]


def test_strip_salts_and_catalysts_keeps_neutral_organometallics():
    # Grignard reagent and a boronic acid genuinely contribute atoms to the product.
    assert strip_salts_and_catalysts(["Br[Mg]c1ccccc1", "OB(O)c1cccs1"]) == [
        "Br[Mg]c1ccccc1",
        "OB(O)c1cccs1",
    ]


def test_strip_salts_and_catalysts_drops_charged_carbonate_despite_carbon():
    assert strip_salts_and_catalysts(["CCO", "O=C([O-])[O-]", "[Na+]", "[Na+]"]) == ["CCO"]


def test_strip_salts_and_catalysts_keeps_unparseable_fragments():
    assert strip_salts_and_catalysts(["not-a-smiles"]) == ["not-a-smiles"]


def test_is_core_exact_match_ignores_auxiliary_salts_on_both_sides():
    predicted = ["CCN(CC)c1ccc(C(=O)c2ccccc2C(=O)O)c(C)c1", "CN(C)c1cccc(N(C)C)c1"]
    reference = [
        "CC(=O)OC(C)=O",
        "CCN(CC)c1ccc(C(=O)c2ccccc2C(=O)O)c(C)c1",
        "CN(C)c1cccc(N(C)C)c1",
        "[Na+]",
        "[OH-]",
    ]
    assert is_exact_match(predicted, reference) is False
    assert is_core_exact_match(predicted, reference) is False  # CC(=O)OC(C)=O is organic, kept


def test_is_core_exact_match_true_once_only_salts_differ():
    predicted = ["CCO", "[Na+]"]
    reference = ["CCO", "[Na+]", "[OH-]"]
    assert is_exact_match(predicted, reference) is False
    assert is_core_exact_match(predicted, reference) is True


def test_is_core_exact_match_smiles_false_for_empty_prediction_or_reference():
    assert is_core_exact_match_smiles("", "CCO") is False
    assert is_core_exact_match_smiles("CCO", None) is False
