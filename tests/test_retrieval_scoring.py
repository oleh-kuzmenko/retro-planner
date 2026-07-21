import pytest

from retro_planner.chemistry import tanimoto_similarity
from retro_planner.config import DEFAULT_RETRIEVAL_WEIGHTS, EXPERIMENTAL_RETRIEVAL_WEIGHTS
from retro_planner.retrieval import merge_retrieval_hits


def test_tanimoto_similarity_identical_vectors():
    vector = [1, 0, 1, 1, 0]
    assert tanimoto_similarity(vector, vector) == 1.0


def test_tanimoto_similarity_disjoint_vectors():
    a = [1, 1, 0, 0]
    b = [0, 0, 1, 1]
    assert tanimoto_similarity(a, b) == 0.0


def test_tanimoto_similarity_partial_overlap():
    a = [1, 1, 1, 0, 0]
    b = [1, 1, 0, 1, 0]
    # intersection = bits {0,1} = 2, union = bits {0,1,2,3} = 4
    assert tanimoto_similarity(a, b) == pytest.approx(0.5)


def test_tanimoto_similarity_all_zero_vectors_returns_zero():
    assert tanimoto_similarity([0, 0, 0], [0, 0, 0]) == 0.0


def test_default_retrieval_weights_are_two_component():
    assert DEFAULT_RETRIEVAL_WEIGHTS.reaction_class == 0.0
    assert DEFAULT_RETRIEVAL_WEIGHTS.molecule + DEFAULT_RETRIEVAL_WEIGHTS.reaction == pytest.approx(1.0)


def test_merge_retrieval_hits_default_formula_ignores_reaction_class():
    molecule_hits = [
        {"reaction_id": "r1", "molecule_similarity": 0.8, "reaction_class": "acylation"},
    ]
    transform_hits = [
        {"reaction_id": "r1", "reaction_similarity": 0.4, "reaction_class": "acylation"},
    ]
    target_classes = {"acylation"}

    merged = merge_retrieval_hits(molecule_hits, transform_hits, target_classes)

    assert len(merged) == 1
    reaction = merged[0]
    # class similarity is still computed and exposed for the UI/experimental path...
    assert reaction["reaction_class_similarity"] == 1.0
    # ...but the default weight is 0.0, so it must not affect the final score.
    assert reaction["final_hybrid_score"] == pytest.approx(
        DEFAULT_RETRIEVAL_WEIGHTS.molecule * 0.8 + DEFAULT_RETRIEVAL_WEIGHTS.reaction * 0.4
    )


def test_merge_retrieval_hits_experimental_weights_include_class_term():
    molecule_hits = [
        {"reaction_id": "r1", "molecule_similarity": 0.8, "reaction_class": "acylation"},
    ]
    transform_hits = [
        {"reaction_id": "r1", "reaction_similarity": 0.4, "reaction_class": "acylation"},
    ]
    target_classes = {"acylation"}

    merged = merge_retrieval_hits(
        molecule_hits,
        transform_hits,
        target_classes,
        weights=EXPERIMENTAL_RETRIEVAL_WEIGHTS,
    )

    reaction = merged[0]
    expected = (
        EXPERIMENTAL_RETRIEVAL_WEIGHTS.molecule * 0.8
        + EXPERIMENTAL_RETRIEVAL_WEIGHTS.reaction * 0.4
        + EXPERIMENTAL_RETRIEVAL_WEIGHTS.reaction_class * 1.0
    )
    assert reaction["final_hybrid_score"] == pytest.approx(expected)


def test_merge_retrieval_hits_sorts_by_final_score_descending():
    molecule_hits = [
        {"reaction_id": "low", "molecule_similarity": 0.1},
        {"reaction_id": "high", "molecule_similarity": 0.9},
    ]

    merged = merge_retrieval_hits(molecule_hits, [], set())

    assert [reaction["reaction_id"] for reaction in merged] == ["high", "low"]


def test_merge_retrieval_hits_merges_molecule_and_transform_by_reaction_id():
    molecule_hits = [{"reaction_id": "r1", "molecule_similarity": 0.6}]
    transform_hits = [{"reaction_id": "r1", "reaction_similarity": 0.2}]

    merged = merge_retrieval_hits(molecule_hits, transform_hits, set())

    assert len(merged) == 1
    assert merged[0]["molecule_similarity"] == 0.6
    assert merged[0]["reaction_similarity"] == 0.2
