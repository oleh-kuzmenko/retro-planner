#!/usr/bin/env python3
"""Assemble the end-to-end cascade demonstration inputs (thesis subsection 3.7).

Picks demonstration targets from the ORD eval set by a deterministic rule -- the
first N records that have at least one reference condition field and stay small
enough to be legible as a printed structural formula -- so the choice of examples
cannot be read as selected by outcome. The size limits look only at the reference
record (product size, reactant size, fragment count, length of the reference
catalyst string); none of them depends on what Model 1 predicted. For each target
it takes the Model 1 candidates already generated for variant 2
(`v2_refixed_ord_topk.json`), applies the inter-stage RDKit validity filter
described in subsection 2.1, and emits the rows Model 2 is then run on.

Example:
    python scripts/models/demo_cascade_inputs.py \\
        --eval-targets data/v2_ord_eval_targets.json \\
        --model1-topk experiments/v2_model1_topk/v2_refixed_ord_topk.json \\
        --output experiments/v2_model1_topk/demo_cascade_inputs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

CONDITION_FIELDS = ("solvent", "catalyst", "temperature_celsius", "yield_percent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-targets", type=Path, default=Path("data/v2_ord_eval_targets.json"))
    parser.add_argument("--model1-topk", type=Path, default=Path("experiments/v2_model1_topk/v2_refixed_ord_topk.json"))
    parser.add_argument("--count", type=int, default=5, help="How many demonstration targets to take.")
    parser.add_argument("--max-product-heavy", type=int, default=16, help="Heavy-atom cap on the target product.")
    parser.add_argument("--max-reactant-heavy", type=int, default=22, help="Heavy-atom cap on the whole reference reactant set.")
    parser.add_argument("--max-fragments", type=int, default=3, help="Cap on the number of components in a reactant set.")
    parser.add_argument("--max-catalyst-chars", type=int, default=25, help="Cap on the length of the reference catalyst SMILES.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def has_condition(record: dict) -> bool:
    return any(record.get(field) not in (None, "") for field in CONDITION_FIELDS)


def heavy_atoms(smiles: str) -> int:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    return molecule.GetNumHeavyAtoms() if molecule is not None else 10**6


def is_legible(target: dict, args: argparse.Namespace, split_fragments) -> bool:
    """Size limits that keep a demonstration reaction readable when printed.

    Every check reads the reference record only, so the filter stays independent
    of what Model 1 predicted for that target."""
    reference = target["reactants_smiles"]
    catalyst = target.get("catalyst")
    return (
        heavy_atoms(target["product_smiles"]) <= args.max_product_heavy
        and heavy_atoms(reference) <= args.max_reactant_heavy
        and len(split_fragments(reference)) <= args.max_fragments
        and (not catalyst or len(catalyst) <= args.max_catalyst_chars)
    )


def main() -> None:
    from retro_eval.evaluation import is_exact_match_smiles, is_valid_smiles, split_fragments, strip_salts_and_catalysts

    args = parse_args()

    targets = json.loads(args.eval_targets.read_text(encoding="utf-8"))
    predictions = json.loads(args.model1_topk.read_text(encoding="utf-8"))["records"]
    by_product = {record["product_smiles"]: record for record in predictions}

    rows = []
    for target in targets:
        if len(rows) == args.count:
            break
        if not has_condition(target):
            continue
        if not is_legible(target, args, split_fragments):
            continue
        prediction = by_product.get(target["product_smiles"])
        if prediction is None:
            continue

        valid = [c for c in prediction["candidates"] if is_valid_smiles(c)]
        if not valid:
            continue
        top1 = valid[0]

        reference = target["reactants_smiles"]
        core_match = is_exact_match_smiles(
            ".".join(strip_salts_and_catalysts(split_fragments(top1))),
            ".".join(strip_salts_and_catalysts(split_fragments(reference))),
        )

        rows.append(
            {
                "reaction_id": target["reaction_id"],
                "product_smiles": target["product_smiles"],
                "reactants_smiles": top1,
                "reference_reactants_smiles": reference,
                "candidates_generated": len(prediction["candidates"]),
                "candidates_valid": len(valid),
                "exact_match_top1": is_exact_match_smiles(top1, reference),
                "core_exact_match_top1": core_match,
                "reference_conditions": {field: target.get(field) for field in CONDITION_FIELDS},
            }
        )

    if len(rows) < args.count:
        raise SystemExit(f"only {len(rows)} usable targets found, wanted {args.count}")

    args.output.write_text(json.dumps({"records": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} demonstration rows to {args.output}")
    for row in rows:
        print(f"  {row['reaction_id']}  exact={row['exact_match_top1']}  core={row['core_exact_match_top1']}")


if __name__ == "__main__":
    main()
