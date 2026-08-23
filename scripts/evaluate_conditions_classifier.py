#!/usr/bin/env python3
"""Score a `train_conditions_classifier.py` checkpoint on the Model 2 test set.

The heads rank their whole class list, so top-k is a slice of the softmax: k different answers,
never the same substance twice. Beam search had to earn that and did not -- five beams of the
generative runs carry 2.84 distinct solvents.

The output file has the shape `evaluate_conditions_model_topk.py` writes, with each rank
serialized into the same compact target string, and the summary is computed by
`reaggregate_conditions_topk.recompute` -- the same function that reproduces the generative
runs' own numbers. Nothing about the metric is reimplemented here, so the two families of runs
stay directly comparable and `mcnemar_conditions.py` pairs them without a special case.

Example:
    python scripts/evaluate_conditions_classifier.py \\
        --model-dir /kaggle/working/e1_classifier/final \\
        --test-file data/v2_ord_roles_full/conditions_test_clean.jsonl \\
        --top-k 5 --output experiments/v2_model2_roles/E1_classifier_clean_topk.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from train_conditions_classifier import ABSENT, NUMERIC_FIELDS, load_classifier
from train_conditions_model import format_input, format_target
from reaggregate_conditions_topk import recompute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N rows.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def class_text(field: str, name) -> str | None:
    """Turn a class back into the text a generated candidate would have carried."""
    if name == ABSENT:
        return None
    if field in NUMERIC_FIELDS:
        return f"{float(name):g}"
    return str(name)


def main() -> None:
    args = parse_args()
    model, tokenizer, meta = load_classifier(args.model_dir)
    model.eval().to(args.device)

    fields = tuple(meta["fields"])
    reactants_field = meta["reactants_field"]
    rows = [json.loads(line) for line in args.test_file.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} test rows | fields {list(fields)} | reactant side {reactants_field}")

    texts = [format_input(row, "", reactants_field) for row in rows]
    ranked: dict[str, list[list[str | None]]] = {field: [] for field in fields}

    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(
                texts[start : start + args.batch_size],
                max_length=meta["max_source_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(args.device)
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            for field in fields:
                logits = output[f"logits_{field}"]
                k = min(args.top_k, logits.shape[-1])
                top = torch.topk(logits, k=k, dim=-1).indices.tolist()
                classes = meta["vocabularies"][field]["classes"]
                ranked[field].extend([[class_text(field, classes[i]) for i in row] for row in top])

    records = []
    for position, row in enumerate(rows):
        candidates = [
            format_target(
                {field: ranked[field][position][rank] for field in fields}, "compact", fields
            )
            for rank in range(args.top_k)
        ]
        records.append(
            {
                "product_smiles": row["product_smiles"],
                "reference": row,
                "candidates_raw": candidates,
            }
        )

    data = {
        "summary": {
            "total": len(records),
            "num_beams": args.top_k,
            "target_format": "compact",
            "condition_fields": list(fields),
            "reactants_field": reactants_field,
        },
        "records": records,
    }
    summary = recompute(data)
    summary["model_kind"] = "encoder_classifier"
    summary["base_model"] = meta["base_model"]
    summary["class_counts"] = {field: len(meta["vocabularies"][field]["classes"]) for field in fields}
    data["summary"] = summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    for field in fields:
        for kind in ("exact_match", "same_group", "within_tol", "same_bucket", "record_level"):
            key = f"{field}_{kind}_top{args.top_k}"
            if summary.get(key) is not None:
                print(f"  {key}: {summary[key] * 100:.1f}%")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
