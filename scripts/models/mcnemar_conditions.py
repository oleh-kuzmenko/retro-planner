#!/usr/bin/env python3
"""Paired McNemar test between two Model 2 top-k result files, field by field.

`mcnemar_topk.py` does this for Model 1, where a record is one prediction. Model 2
answers several fields per record and each one carries its own denominator: the strict
scale scores only the records whose reference has the field, so solvent, catalyst and
temperature are three different paired samples over the same 5,687 test rows.

Records are paired by position and the pairing is asserted on `product_smiles`. Only the
records both runs can be scored on enter a field's test, which for the strict scale means
the ones whose reference names the field.

Example:
    python scripts/models/mcnemar_conditions.py \\
        experiments/v2_model2_roles/B2_conditions_compoundt5_3f_clean_topk.json \\
        experiments/v2_model2_roles/B2_conditions_t5small_3f_clean_topk.json \\
        "B2: CompoundT5 -> t5-small"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_conditions_model import (
    NUMERIC_BUCKET_EDGES,
    TEMPERATURE_TOLERANCE_C,
    YIELD_TOLERANCE_PCT,
    normalize_components,
    same_bucket,
    to_number,
)
from evaluate_conditions_model_topk import decode_and_parse
from mcnemar_topk import exact_binomial_two_sided

STRING_FIELDS = ("reagents", "solvent", "catalyst")
NUMERIC_FIELDS = ("temperature_celsius", "yield_percent")
TOLERANCES = {
    "temperature_celsius": TEMPERATURE_TOLERANCE_C,
    "yield_percent": YIELD_TOLERANCE_PCT,
}


def hits(path: Path, field: str, k: int) -> list[bool | None]:
    """Per-record correctness on the strict scale; None where the field has no reference
    value and the strict scale therefore does not score the record."""
    data = json.loads(path.read_text(encoding="utf-8"))
    target_format = data["summary"].get("target_format", "json")
    fields = tuple(data["summary"]["condition_fields"])
    out: list[bool | None] = []
    for record in data["records"]:
        row = record["reference"]
        parsed = [decode_and_parse(c, target_format, fields) for c in record["candidates_raw"][:k]]
        if field in STRING_FIELDS:
            reference = normalize_components(row.get(field))
            if reference is None:
                out.append(None)
                continue
            out.append(
                any(p is not None and normalize_components(p.get(field)) == reference for p in parsed)
            )
        else:
            reference = to_number(row.get(field))
            if reference is None:
                out.append(None)
                continue
            tolerance = TOLERANCES[field]
            hit = False
            for p in parsed:
                if p is None:
                    continue
                predicted = to_number(p.get(field))
                if predicted is not None and abs(predicted - reference) <= tolerance:
                    hit = True
                    break
            out.append(hit)
    return out


def products(path: Path) -> list[str]:
    return [r["product_smiles"] for r in json.loads(path.read_text(encoding="utf-8"))["records"]]


def shared_fields(baseline: Path, variant: Path) -> list[str]:
    def declared(path: Path) -> list[str]:
        return json.loads(path.read_text(encoding="utf-8"))["summary"]["condition_fields"]

    both = set(declared(baseline)) & set(declared(variant))
    return [f for f in STRING_FIELDS + NUMERIC_FIELDS if f in both]


def main() -> None:
    baseline_path, variant_path = Path(sys.argv[1]), Path(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else f"{baseline_path.stem} -> {variant_path.stem}"
    assert products(baseline_path) == products(variant_path), "record order differs"

    print(f"=== {label}")
    for field in shared_fields(baseline_path, variant_path):
        for k in (1, 3, 5):
            left = hits(baseline_path, field, k)
            right = hits(variant_path, field, k)
            scored = [(x, y) for x, y in zip(left, right) if x is not None and y is not None]
            b = sum(1 for x, y in scored if x and not y)  # baseline only
            c = sum(1 for x, y in scored if y and not x)  # variant only
            p = exact_binomial_two_sided(b, c)
            flag = "  *" if p < 0.05 else ""
            share = lambda side: sum(1 for pair in scored if pair[side]) / len(scored) * 100
            print(
                f"  {field:20s} top{k}  n={len(scored):5d}  "
                f"{share(0):5.1f}% -> {share(1):5.1f}%   discordant {b:4d}/{c:4d}   p={p:.2e}{flag}"
            )


if __name__ == "__main__":
    main()
