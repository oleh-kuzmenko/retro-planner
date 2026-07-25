import json

from retro_eval.harness.records import load_records


def test_load_records_uspto_format_has_no_ord_fields(tmp_path):
    path = tmp_path / "uspto.json"
    path.write_text(
        json.dumps(
            [
                {
                    "product_smiles": "CC(C)(C)OC(=O)N1CCC(NC(=O)c2ccccc2)CC1",
                    "reactants_smiles": "CC(C)(C)OC(=O)N1CCC(N)CC1.O=C(Cl)c1ccccc1",
                }
            ]
        )
    )

    records = load_records(path)

    assert len(records) == 1
    assert records[0].index == 0
    assert records[0].reactants_smiles == "CC(C)(C)OC(=O)N1CCC(N)CC1.O=C(Cl)c1ccccc1"
    assert records[0].is_ord is False


def test_load_records_ord_format_sets_is_ord_true(tmp_path):
    path = tmp_path / "ord.json"
    path.write_text(
        json.dumps(
            [
                {
                    "reaction_id": "ord-e986ebcd9edb45428124e738770f3eaa",
                    "product_smiles": "O=C(/C=C/c1ccccc1)c1ccccc1",
                    "reactants_smiles": "CC(=O)c1ccccc1.O=Cc1ccccc1",
                    "solvent": "CCO, O",
                    "temperature_celsius": 25.0,
                    "catalyst": None,
                    "yield_percent": 92.0,
                }
            ]
        )
    )

    records = load_records(path)

    assert len(records) == 1
    assert records[0].reaction_id == "ord-e986ebcd9edb45428124e738770f3eaa"
    assert records[0].temperature_celsius == 25.0
    assert records[0].is_ord is True


def test_load_records_skips_missing_product_smiles_but_keeps_original_index(tmp_path):
    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps(
            [
                {"product_smiles": "CCO", "reactants_smiles": "CC.O"},
                {"reactants_smiles": "CCN"},
                {"product_smiles": "CCN", "reactants_smiles": "CC.N"},
            ]
        )
    )

    records = load_records(path)

    assert [record.index for record in records] == [0, 2]
