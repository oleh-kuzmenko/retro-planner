from scripts.index_uspto50k_to_qdrant import normalize_row


def test_normalize_row_supports_reaction_smiles_schema():
    row = {
        "reaction_smiles": "CCO>>CC=O",
        "class": "8",
    }

    normalized = normalize_row(row, "train", 7)

    assert normalized is not None
    assert normalized["reaction_id"] == "uspto50k_train_7"
    assert normalized["reactants_smiles"] == "CCO"
    assert normalized["product_smiles"] == "CC=O"
    assert normalized["reaction_class_normalized"] == "oxidation"


def test_normalize_row_supports_reactants_and_product_schema():
    row = {
        "id": "rxn-1",
        "reactants": "CCO",
        "product": "CC=O",
        "reaction_class": "oxidation",
    }

    normalized = normalize_row(row, "test", 1)

    assert normalized is not None
    assert normalized["reaction_id"] == "rxn-1"
    assert normalized["reaction_smiles"] == "CCO>>CC=O"


def test_normalize_row_skips_rows_without_reactants_or_product():
    assert normalize_row({"product": "CCO"}, "train", 1) is None
