#!/usr/bin/env python3
"""Paired McNemar test between two top-k result files, matching the scoring in
`scripts/models/run_reactiont5_topk.py` (same is_exact_match / strip_salts_and_catalysts calls).

Records are paired by position; both files are written from the same eval-targets JSON,
so index i is the same product in both. Verified by comparing product_smiles.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from retro_eval.evaluation import is_exact_match, split_fragments, strip_salts_and_catalysts


def hits(path: Path, k: int, core: bool) -> list[bool]:
    data = json.loads(path.read_text())
    out = []
    for record in data["records"]:
        reference = split_fragments(record["reactants_smiles"])
        if core:
            reference = strip_salts_and_catalysts(reference)
        hit = False
        for candidate in record["candidates"][:k]:
            fragments = split_fragments(candidate)
            if core:
                fragments = strip_salts_and_catalysts(fragments)
            if is_exact_match(fragments, reference):
                hit = True
                break
        out.append(hit)
    return out


def products(path: Path) -> list[str]:
    return [r["product_smiles"] for r in json.loads(path.read_text())["records"]]


def exact_binomial_two_sided(b: int, c: int) -> float:
    """Exact McNemar (binomial) p-value. The chi-square approximation is unreliable
    at the discordant counts these 300-record sets produce."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def main() -> None:
    baseline_path, variant_path, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    assert products(baseline_path) == products(variant_path), "record order differs"

    print(f"=== {label}  (n={len(products(baseline_path))})")
    for core in (False, True):
        for k in (1, 3, 5):
            a = hits(baseline_path, k, core)
            b_ = hits(variant_path, k, core)
            b = sum(1 for x, y in zip(a, b_) if x and not y)  # baseline only
            c = sum(1 for x, y in zip(a, b_) if y and not x)  # variant only
            p = exact_binomial_two_sided(b, c)
            name = ("core" if core else "exact") + f"_top{k}"
            flag = "  *" if p < 0.05 else ""
            print(
                f"  {name:12s} {sum(a)/len(a)*100:5.1f}% -> {sum(b_)/len(b_)*100:5.1f}%"
                f"   discordant {b:3d}/{c:3d}   p={p:.4f}{flag}"
            )


if __name__ == "__main__":
    main()
