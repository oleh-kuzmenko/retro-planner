import sys

from scripts import index_uspto50k_to_qdrant as indexer
from scripts.index_uspto50k_to_qdrant import (
    DEFAULT_INDEX_LIMIT,
    normalize_ord_reaction,
    normalize_row,
    parse_args,
    remaining_index_limit,
    require_optional_dependencies,
)


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


def test_normalize_row_keeps_isolated_hydrogen_fragments():
    row = {
        "reaction_smiles": "[H].CC>>CCO",
        "class": "1",
    }

    normalized = normalize_row(row, "train", 8)

    assert normalized is not None
    assert "[H]" in normalized["reactants_smiles"]


def test_normalize_ord_reaction_extracts_conditions():
    reaction = {
        "reaction_id": "ord-rxn-1",
        "inputs": {
            "reactant": {
                "components": [
                    {
                        "reaction_role": "REACTANT",
                        "identifiers": [
                            {"type": "SMILES", "value": "CCO"},
                        ],
                    },
                    {
                        "reaction_role": "SOLVENT",
                        "identifiers": [
                            {"type": "SMILES", "value": "O"},
                        ],
                    },
                    {
                        "reaction_role": "CATALYST",
                        "identifiers": [
                            {"type": "NAME", "value": "Pd/C"},
                        ],
                    },
                ],
            },
        },
        "conditions": {
            "temperature": {
                "setpoint": {"value": 298.15, "units": "KELVIN"},
            },
        },
        "outcomes": [
            {
                "products": [
                    {
                        "identifiers": [
                            {"type": "SMILES", "value": "CC=O"},
                        ],
                        "measurements": [
                            {"type": "YIELD", "percentage": {"value": 82}},
                        ],
                    },
                ],
            },
        ],
    }

    normalized = normalize_ord_reaction(reaction, "sample.pb.gz", 0)

    assert normalized is not None
    assert normalized["source"] == "ORD"
    assert normalized["reactants_smiles"] == "CCO"
    assert normalized["product_smiles"] == "CC=O"
    assert normalized["conditions"]["solvents"] == ["O"]
    assert normalized["conditions"]["catalysts"] == ["Pd/C"]
    assert normalized["temperature_celsius"] == 25.0
    assert normalized["yield_percent"] == 82.0


def test_require_optional_dependencies_skips_ord_modules_for_uspto_only(monkeypatch):
    checked = []

    def fake_find_spec(module):
        checked.append(module)
        return object()

    monkeypatch.setattr(indexer.importlib.util, "find_spec", fake_find_spec)

    require_optional_dependencies(["uspto"], None)

    assert "datasets" in checked
    assert "tqdm" in checked
    assert "ord_schema" not in checked
    assert "huggingface_hub" not in checked


def test_require_optional_dependencies_fails_before_ord_download(monkeypatch):
    def fake_find_spec(module):
        if module == "ord_schema":
            return None
        return object()

    monkeypatch.setattr(indexer.importlib.util, "find_spec", fake_find_spec)

    try:
        require_optional_dependencies(["ord"], None)
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing ord_schema to stop indexing.")

    assert "ord_schema" in message
    assert 'pip install -e ".[indexing]"' in message


def test_parse_args_defaults_to_500k_indexed_reactions(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["indexer"])

    args = parse_args()

    assert args.limit == DEFAULT_INDEX_LIMIT


def test_parse_args_limit_zero_means_unlimited(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["indexer", "--limit", "0"])

    args = parse_args()

    assert args.limit is None


def test_remaining_index_limit_tracks_total_cap():
    assert remaining_index_limit(500_000, 49_999) == 450_001
    assert remaining_index_limit(500_000, 500_000) == 0
    assert remaining_index_limit(None, 500_000) is None
