#!/usr/bin/env python3
"""Merge Model 2 result files scored on consecutive chunks of one test file.

`evaluate_conditions_model_topk.py` writes nothing until the last record is scored, and a
Colab GPU session ends without warning after about an hour -- a 5,687-record run once died
at 5,600 and cost the whole hour. Scoring the test in chunks caps that loss at one chunk;
this puts the chunks back together and recomputes the summary from the merged generations
with `reaggregate_conditions_topk`, the same code that reproduces the evaluator's numbers
on a file it wrote itself.

The chunks must come from the same run: the merge refuses to join files whose evaluation
settings differ, since a summary over a mixture of beam widths or checkpoints means nothing.

Example:
    python scripts/models/merge_conditions_topk.py \\
        --output experiments/v2_model2_roles/cascade_m3_clean_topk.json \\
        chunk_aa.json chunk_ab.json chunk_ac.json chunk_ad.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reaggregate_conditions_topk import recompute

# Keys that describe how the run was made rather than what it scored; they must agree.
SETTING_KEYS = ("num_beams", "target_format", "condition_fields", "always_fields", "reactants_field", "model_dir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chunks", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged: list[dict] = []
    settings = None
    summary = None
    seen: set[str] = set()
    for path in args.chunks:
        data = json.loads(path.read_text(encoding="utf-8"))
        here = {key: data["summary"].get(key) for key in SETTING_KEYS}
        if settings is None:
            settings, summary = here, dict(data["summary"])
        elif here != settings:
            raise SystemExit(f"{path.name} was scored with different settings: {here} != {settings}")
        for record in data["records"]:
            product = record["product_smiles"]
            if product in seen:
                continue  # a resumed chunk can overlap the previous one
            seen.add(product)
            merged.append(record)
        print(f"{path.name}: {len(data['records'])} record(s)")

    out = {"summary": summary, "records": merged}
    out["summary"] = recompute(out)
    args.output.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{args.output}: {len(merged)} record(s) merged")


if __name__ == "__main__":
    main()
