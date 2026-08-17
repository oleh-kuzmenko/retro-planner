#!/usr/bin/env python3
"""Paired comparison of the three roles-format runs on the same 5,687-record clean test.

Two questions the summary files cannot answer on their own:
  1. is T2's temperature gain over T1 larger than paired noise;
  2. is T2's solvent deficit against B3 confined to the records that carry no temperature,
     which would say the temperature filter shifted the population rather than the model.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from scipy.stats import binomtest

from evaluate_conditions_model_topk import decode_and_parse
from evaluate_conditions_model import normalize_components, to_number

RESULTS_DIR = ROOT / "experiments" / "v2_model2_roles"
RUNS = {
    "B3": RESULTS_DIR / "B3_conditions_roles_clean_topk.json",
    "T1": RESULTS_DIR / "T1_conditions_temp38k_clean_topk.json",
    "T2": RESULTS_DIR / "T2_conditions_temp141k_clean_topk.json",
}
FIELDS = ("reagents", "solvent", "catalyst", "temperature_celsius")
K = 5
TOL = 10.0


def hits(path):
    """Per-record top-5 correctness vectors, keyed by test-row index."""
    data = json.load(open(path))
    out = {"solvent": [], "catalyst": [], "reagents": [], "temperature_celsius": [], "has_temp": []}
    for rec in data["records"]:
        ref = rec["reference"]
        parsed = [decode_and_parse(c, "compact", FIELDS) for c in rec["candidates_raw"][:K]]
        for field in ("reagents", "solvent", "catalyst"):
            ref_set = normalize_components(ref.get(field))
            if ref_set is None:
                out[field].append(None)
                continue
            out[field].append(any(p is not None and normalize_components(p.get(field)) == ref_set for p in parsed))
        ref_t = to_number(ref.get("temperature_celsius"))
        out["has_temp"].append(ref_t is not None)
        if ref_t is None:
            out["temperature_celsius"].append(None)
        else:
            preds = [to_number(p.get("temperature_celsius")) for p in parsed if p is not None]
            preds = [v for v in preds if v is not None]
            out["temperature_celsius"].append(any(abs(v - ref_t) <= TOL for v in preds))
    return out


def mcnemar(a, b, label):
    """Exact binomial test on the records where exactly one run is right."""
    only_a = sum(1 for x, y in zip(a, b) if x is True and y is False)
    only_b = sum(1 for x, y in zip(a, b) if x is False and y is True)
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    print(f"  {label}: {only_a} vs {only_b} discordant, p = {p:.2e}")


H = {name: hits(path) for name, path in RUNS.items()}

print("temperature +/-10 C top-5, paired:")
mcnemar(H["T2"]["temperature_celsius"], H["T1"]["temperature_celsius"], "T2 wins / T1 wins")
mcnemar(H["T2"]["temperature_celsius"], H["B3"]["temperature_celsius"], "T2 wins / B3 wins")

print("\nsolvent and catalyst top-5 split by whether the test record carries a temperature:")
for field in ("solvent", "catalyst", "reagents"):
    for subset, want in (("with temp", True), ("no temp", False)):
        line = [f"  {field:9} {subset:9}"]
        for name in ("B3", "T2"):
            h = H[name]
            vals = [v for v, t in zip(h[field], h["has_temp"]) if t is want and v is not None]
            line.append(f"{name} {sum(vals) / len(vals):.3f} (n={len(vals)})")
        print("  ".join(line))
