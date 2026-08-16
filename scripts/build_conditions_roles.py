#!/usr/bin/env python3
"""Split ORD's reactant side into substrates and reagents, for Model 2's B3 configuration.

Two defects in the cascade are fixed by the same data transform.

**The interface.** Model 2 trained on ORD sees an input averaging 3.46 molecules, because
ORD writes the whole vessel charge. At inference it is handed Model 1's output, which
under USPTO's convention averages 1.67. Training and inference therefore saw different
input distributions -- a mismatch that only appears once the best Model 1 is USPTO-trained,
and that was invisible while both models were ORD-trained. Writing `reactants_smiles` as
substrates only makes the training input the same shape as the served one.

**The third role.** A reaction needs substrates, conditions, and the stoichiometric
reagents and bases in between -- K2CO3, NaH, EDC. The first went to Model 1 and the second
to Model 2; the third was predicted by neither, living in ORD's reactant field and absent
from Model 2's targets. Only 0.7% of those species are duplicated in the solvent or
catalyst fields, so nothing else in the record covers them. `reagents` becomes a target
field, and the cascade finally accounts for every substance in the record.

Roles come from `scripts/models/substrate_role_analysis.py`, whose rule reproduces
USPTO-50K's own convention to within 1.6% of fragments.

Example:
    python scripts/build_conditions_roles.py \
        --input data/v2_ord_train_300k/conditions_train.jsonl \
        --output data/v2_ord_roles/conditions_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from substrate_role_analysis import canonical, is_substrate

LOGGER = logging.getLogger("retro_eval.build_conditions_roles")

# ", " is what ORD already uses to join multiple solvents or catalysts, and what
# `normalize_components` splits on, so the new field needs no separate parsing path.
COMPONENT_SEPARATOR = ", "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker processes. The cost is one MCS per (fragment, product) pair, and pairs "
        "almost never repeat across records, so the in-process cache does not help and a "
        "single worker runs at ~13 rows/s -- three hours for the 138,869-row training set.",
    )
    parser.add_argument(
        "--drop-empty-substrates",
        action="store_true",
        help="Skip records where the rule leaves no substrate at all. Off by default: such "
        "a record still carries a usable conditions target, and dropping it would bias the "
        "training set toward reactions the rule happens to understand.",
    )
    return parser.parse_args()


def split_roles(reactants: str, product: str) -> tuple[list[str], list[str]]:
    substrates: list[str] = []
    reagents: list[str] = []
    for fragment in reactants.split("."):
        if not fragment:
            continue
        canon = canonical(fragment)
        if canon is None:
            continue
        target = substrates if is_substrate(canon, product) else reagents
        if canon not in target:
            target.append(canon)
    return substrates, reagents


def annotate(row: dict) -> dict | None:
    product = canonical(row["product_smiles"])
    if product is None:
        return None
    substrates, reagents = split_roles(row["reactants_smiles"], product)
    out = dict(row)
    out["product_smiles"] = product
    # Model 1 hands over substrates only, so this is what Model 2 must train on.
    out["reactants_smiles"] = ".".join(substrates)
    out["reagents"] = COMPONENT_SEPARATOR.join(reagents) if reagents else None
    out["full_reactants_smiles"] = row["reactants_smiles"]
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    LOGGER.info("Read %d row(s) from %s | jobs=%d", len(rows), args.input, args.jobs)

    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            annotated = pool.map(annotate, rows, chunksize=64)
    else:
        annotated = [annotate(row) for row in rows]

    written = 0
    no_substrate = 0
    no_reagent = 0
    substrate_total = 0
    reagent_total = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for out in annotated:
            if out is None:
                continue
            substrates = [f for f in out["reactants_smiles"].split(".") if f]
            reagents = out["reagents"].split(COMPONENT_SEPARATOR) if out["reagents"] else []
            if not substrates:
                no_substrate += 1
                if args.drop_empty_substrates:
                    continue
            if not reagents:
                no_reagent += 1
            substrate_total += len(substrates)
            reagent_total += len(reagents)
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    LOGGER.info("Wrote %d row(s) to %s", written, args.output)
    LOGGER.info(
        "Substrates per record %.2f | reagents per record %.2f | no substrate %d (%.1f%%) | no reagent %d (%.1f%%)",
        substrate_total / written,
        reagent_total / written,
        no_substrate,
        100 * no_substrate / len(rows),
        no_reagent,
        100 * no_reagent / len(rows),
    )


if __name__ == "__main__":
    main()
