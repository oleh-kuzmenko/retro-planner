from retro_planner.evaluation import (
    canonical_precursor_set,
    format_results_table,
    is_exact_match,
    structure_success_rate,
    top_k_exact_match,
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


def test_top_k_exact_match_counts_hit_within_k():
    # Target 0: correct answer is the 2nd-ranked candidate -> hit at k=2, miss at k=1.
    # Target 1: correct answer is the 1st-ranked candidate -> hit at k=1.
    predictions = [
        [["CC(=O)O"], ["CCO", "CC(=O)O"]],
        [["CCN", "CC(=O)Cl"]],
    ]
    references = [
        ["CCO", "CC(=O)O"],
        ["CCN", "CC(=O)Cl"],
    ]

    assert top_k_exact_match(predictions, references, k=1) == 0.5
    assert top_k_exact_match(predictions, references, k=2) == 1.0


def test_top_k_exact_match_empty_dataset_returns_zero():
    assert top_k_exact_match([], [], k=1) == 0.0


def test_top_k_exact_match_raises_on_length_mismatch():
    try:
        top_k_exact_match([[["CCO"]]], [], k=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched lengths")


def test_structure_success_rate_counts_parseable_smiles():
    assert structure_success_rate(["CCO", "not-a-smiles", "CC(=O)O.CCO"]) == 2 / 3


def test_structure_success_rate_treats_empty_string_as_failure():
    assert structure_success_rate(["CCO", ""]) == 0.5


def test_structure_success_rate_empty_list_returns_zero():
    assert structure_success_rate([]) == 0.0


def test_format_results_table_renders_header_and_rows():
    results = {
        "Zero-shot": {
            "num_targets": 10,
            "top_1": 0.2,
            "top_3": 0.4,
            "structure_success_rate": 0.9,
        },
        "RAG+CoT": {
            "num_targets": 10,
            "top_1": 0.5,
            "top_3": 0.7,
            "structure_success_rate": 0.95,
        },
    }

    table = format_results_table(results, k_values=[1, 3])

    assert "| Mode | N | Top-1 | Top-3 | Structure Success Rate |" in table
    assert "| Zero-shot | 10 | 20.0% | 40.0% | 90.0% |" in table
    assert "| RAG+CoT | 10 | 50.0% | 70.0% | 95.0% |" in table
