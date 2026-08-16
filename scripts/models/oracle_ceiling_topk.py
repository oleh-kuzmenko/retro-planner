#!/usr/bin/env python3
"""Oracle ceiling for merging two top-k result files.

A record counts as an oracle hit at k if *either* model has the reference within its own
top-k. That is the best any merge strategy could reach if it always picked the right
model per record, so it bounds what a smarter ranker than plain interleaving could buy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from retro_eval.evaluation import is_exact_match, split_fragments, strip_salts_and_catalysts


def hits(path: Path, k: int, core: bool) -> list[bool]:
    out = []
    for record in json.loads(path.read_text())["records"]:
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


def main() -> None:
    a_path, b_path, ens_path, label = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]

    print(f"=== {label}")
    print(f"{'metric':12s} {'A':>7s} {'B':>7s} {'merged':>7s} {'oracle':>7s} {'headroom':>9s}")
    for core in (False, True):
        for k in (1, 3, 5):
            a = hits(a_path, k, core)
            b = hits(b_path, k, core)
            e = hits(ens_path, k, core)
            n = len(a)
            oracle = sum(1 for x, y in zip(a, b) if x or y) / n * 100
            merged = sum(e) / n * 100
            name = ("core" if core else "exact") + f"_top{k}"
            print(
                f"{name:12s} {sum(a)/n*100:6.1f}% {sum(b)/n*100:6.1f}% {merged:6.1f}%"
                f" {oracle:6.1f}% {oracle - merged:8.1f}"
            )


if __name__ == "__main__":
    main()
