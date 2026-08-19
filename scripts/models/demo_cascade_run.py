#!/usr/bin/env python3
"""Run the full cascade on a handful of reactions, for the thesis demonstration.

Model 1 proposes substrates for a target product, an RDKit validity filter drops
candidates that are not parseable molecules, and Model 2 reads the product together
with the surviving candidates -- all of them or only the best, per --model2-input --
and proposes the conditions. Nothing here is a measurement -- the quantitative results
come from the two evaluation scripts -- so the run is deliberately tiny and stays on
the CPU.

The two stages must agree on how the reactant side is written. A USPTO-trained Model 1
emits substrates only; an ORD-trained one emits ORD's whole vessel charge.
--reference-field picks the reference written in the same convention, so the
demonstration compares the prediction rather than the two sources' recording habits.

The targets are named explicitly rather than picked by score: they are read from the
clean conditions test set by product SMILES, which keeps the choice auditable and
independent of what either model happens to answer.

Example:
    python scripts/models/demo_cascade_run.py \
        --model1-dir /path/to/model1/final \
        --model2-dir experiments/m2_mixed/model2_conditions_mixed/final \
        --test-file data/v2_ord_roles/conditions_test_clean.jsonl \
        --product "CCCCCCCCCCCCN1CCCC1" \
        --reference-field full_reactants_smiles --model2-input best \
        --output assets/demo/demo_cascade_v6.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_conditions_model_topk import decode_and_parse
from train_conditions_model import format_input, full_schema_code, parse_condition_fields

CONDITION_FIELDS = ("reagents", "solvent", "catalyst", "temperature_celsius")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model1-dir", type=Path, required=True)
    parser.add_argument("--model2-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=Path("data/v2_ord_roles/conditions_test_clean.jsonl"))
    parser.add_argument("--product", action="append", required=True, help="Target product SMILES; repeat for several.")
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--keep", type=int, default=3, help="Candidates kept per stage in the written record.")
    parser.add_argument(
        "--reference-field",
        choices=("reactants_smiles", "full_reactants_smiles"),
        default=None,
        help="Which reactant-side record the demonstration shows as the reference. Left unset it "
        "is read from Model 2's format marker, so the reference is written in the convention both "
        "stages already agreed on; otherwise the comparison would show the two sources' recording "
        "habits instead of the prediction.",
    )
    parser.add_argument(
        "--model2-input",
        choices=("merged", "best"),
        default="merged",
        help="What the second stage reads: `merged` concatenates the kept candidate sets, `best` "
        "passes only the top one. `best` keeps the served input the same shape as Model 2 saw in "
        "training; `merged` hands it every surviving hypothesis at once.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def valid_molecule_set(smiles: str) -> bool:
    """The inter-stage filter: every fragment of a candidate must parse as a molecule."""
    from rdkit import Chem

    fragments = [f for f in smiles.split(".") if f]
    return bool(fragments) and all(Chem.MolFromSmiles(f) is not None for f in fragments)


def generate(model, tokenizer, prompts: list[str], num_beams: int, max_length: int) -> list[list[str]]:
    import torch

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_length=max_length,
            early_stopping=True,
        )
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [decoded[i * num_beams : (i + 1) * num_beams] for i in range(len(prompts))]


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = {json.loads(line)["product_smiles"]: json.loads(line) for line in args.test_file.read_text().splitlines() if line.strip()}
    targets = []
    for product in args.product:
        if product not in rows:
            raise SystemExit(f"product not found in {args.test_file}: {product}")
        targets.append(rows[product])

    marker = json.loads((args.model2_dir / "conditions_format.json").read_text())
    reference_field = args.reference_field or marker.get("reactants_field", "reactants_smiles")
    print(f"Еталонний бік: {reference_field}")

    print(f"Модель 1: {args.model1_dir}")
    tokenizer1 = AutoTokenizer.from_pretrained(str(args.model1_dir))
    model1 = AutoModelForSeq2SeqLM.from_pretrained(str(args.model1_dir)).eval()
    stage1 = generate(model1, tokenizer1, [row["product_smiles"] for row in targets], args.num_beams, 256)
    del model1

    records = []
    for row, candidates in zip(targets, stage1):
        seen: list[str] = []
        for candidate in candidates:
            cleaned = candidate.replace(" ", "")
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        valid = [c for c in seen if valid_molecule_set(c)]
        records.append(
            {
                "product_smiles": row["product_smiles"],
                "reference_substrates": row[reference_field],
                "reference_conditions": {field: row.get(field) for field in CONDITION_FIELDS},
                "candidates_generated": len(seen),
                "candidates_valid": len(valid),
                "predicted_substrates": valid[: args.keep],
            }
        )

    print(f"Модель 2: {args.model2_dir}")
    tokenizer2 = AutoTokenizer.from_pretrained(str(args.model2_dir))
    model2 = AutoModelForSeq2SeqLM.from_pretrained(str(args.model2_dir)).eval()
    always = parse_condition_fields(",".join(marker.get("always_fields") or [])) if marker.get("always_fields") else ()
    schema = full_schema_code(always) if always else ""
    fields = parse_condition_fields(",".join(marker["fields"]))

    answered = [r for r in records if r["predicted_substrates"]]
    for record in answered:
        kept = record["predicted_substrates"]
        record["model2_input_substrates"] = kept[0]
        record["model2_input"] = ".".join(kept) if args.model2_input == "merged" else kept[0]

    prompts = [
        format_input({"product_smiles": r["product_smiles"], "reactants_smiles": r["model2_input"]}, schema)
        for r in answered
    ]
    stage2 = generate(model2, tokenizer2, prompts, args.num_beams, 256)

    for record, candidates in zip(answered, stage2):
        parsed = []
        for candidate in candidates:
            value = decode_and_parse(candidate, marker.get("target_format", "compact"), fields)
            if value is not None and value not in parsed:
                parsed.append(value)
        record["predicted_conditions"] = parsed[: args.keep]
        record["predicted_conditions_raw"] = candidates

    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "model1_dir": str(args.model1_dir),
        "model2_dir": str(args.model2_dir),
        "test_file": str(args.test_file),
        "num_beams": args.num_beams,
        "reference_field": reference_field,
        "model2_input_mode": args.model2_input,
        "condition_fields": list(fields),
    }
    args.output.write_text(json.dumps({**provenance, "records": records}, ensure_ascii=False, indent=1))
    print(f"записано {args.output}")

    for record in records:
        print("\nпродукт    :", record["product_smiles"])
        print("  еталон   :", record["reference_substrates"])
        print("  прогноз  :", record["predicted_substrates"][:1])
        print("  умови    :", record.get("predicted_conditions", [{}])[0])
        print("  еталон   :", record["reference_conditions"])


if __name__ == "__main__":
    main()
