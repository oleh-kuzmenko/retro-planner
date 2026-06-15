from retro_planner.chemistry import (
    canonicalize_smiles,
    generate_morgan_fingerprint,
    reaction_transform_vector,
)
from retro_planner.config import VECTOR_SIZE


def test_canonicalize_smiles_returns_canonical_value():
    assert canonicalize_smiles("OC(C)=O") == "CC(=O)O"


def test_canonicalize_smiles_returns_none_for_invalid_input():
    assert canonicalize_smiles("not-a-smiles") is None


def test_morgan_fingerprint_has_configured_size():
    vector = generate_morgan_fingerprint("CCO")

    assert len(vector) == VECTOR_SIZE
    assert any(value > 0 for value in vector)


def test_reaction_transform_vector_has_configured_size():
    vector = reaction_transform_vector("CCO", "CC=O")

    assert vector is not None
    assert len(vector) == VECTOR_SIZE
