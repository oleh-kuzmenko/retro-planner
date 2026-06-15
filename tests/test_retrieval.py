from retro_planner.config import RetrievalWeights
from retro_planner.retrieval import merge_retrieval_hits


def test_merge_retrieval_hits_deduplicates_and_reranks():
    molecule_hits = [
        {
            "reaction_id": "r1",
            "reaction_class": "acylation",
            "molecule_similarity": 0.7,
            "reaction_smiles": "A>>B",
        },
        {
            "reaction_id": "r2",
            "reaction_class": "oxidation",
            "molecule_similarity": 0.95,
            "reaction_smiles": "C>>D",
        },
    ]
    transform_hits = [
        {
            "reaction_id": "r1",
            "reaction_similarity": 0.9,
            "reaction_smiles": "A>>B",
        },
    ]

    merged = merge_retrieval_hits(
        molecule_hits,
        transform_hits,
        {"acylation"},
        RetrievalWeights(molecule=0.5, reaction=0.3, reaction_class=0.2),
    )

    assert len(merged) == 2
    assert merged[0]["reaction_id"] == "r1"
    assert merged[0]["molecule_similarity"] == 0.7
    assert merged[0]["reaction_similarity"] == 0.9
    assert merged[0]["reaction_class_similarity"] == 1.0
    assert merged[0]["final_hybrid_score"] > merged[1]["final_hybrid_score"]
