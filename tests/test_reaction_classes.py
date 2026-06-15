from retro_planner.reaction_classes import (
    infer_target_reaction_classes,
    normalize_reaction_class,
    reaction_class_similarity,
)


def test_normalize_reaction_class_supports_numeric_aliases_and_keywords():
    assert normalize_reaction_class("2") == "acylation"
    assert normalize_reaction_class("Suzuki coupling") == "coupling"
    assert normalize_reaction_class("Functional Group Addition") == "functional_group_addition"


def test_reaction_class_similarity_scores_exact_and_group_matches():
    assert reaction_class_similarity("acylation", {"acylation"}) == 1.0
    assert reaction_class_similarity("esterification", {"acylation"}) == 0.5
    assert reaction_class_similarity("oxidation", {"coupling"}) == 0.0


def test_infer_target_reaction_classes_detects_ester():
    assert "acylation" in infer_target_reaction_classes("CC(=O)OCC")
