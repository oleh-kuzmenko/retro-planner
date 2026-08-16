"""End-to-end cascade demonstration for thesis subsection 3.7 (Kaggle kernel).

Model 1 (variant 2) candidates and the RDKit inter-stage validity filter were already
applied locally by `scripts/models/demo_cascade_inputs.py`; the rows below are its
output verbatim. This kernel runs only the second stage -- Model 2, fine-tuned on the
138 869-reaction deduplicated pool -- on those product/reactant pairs, with the same
prompt format, beam count and JSON-decoding workaround as
`scripts/evaluate_conditions_model_topk.py`.

Attach the checkpoint dataset `kuzmenkooleh/retro-planner-model2-conditions-t5small-300k-ckpt`.
Writes /kaggle/working/demo_cascade_out.json.
"""

import glob
import json
import os
import shutil

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NUM_BEAMS = 5
MAX_TARGET_LENGTH = 128

RECORDS = [
    {
        "reaction_id": "ord-b82f52deb4634a4bba0b118b41c64e02",
        "product_smiles": "O=P(O)(O)CC(F)=C(F)F",
        "reactants_smiles": "CCOP(=O)(CC(F)=C(F)F)OCC.C[Si](C)(C)Br",
        "reference_conditions": {"solvent": "CO", "catalyst": None, "temperature_celsius": None, "yield_percent": 45.8},
    },
    {
        "reaction_id": "ord-68830d19234b4026807695ab721a68e9",
        "product_smiles": "CN(C)C(Cl)=[N+](C)C.[Cl-]",
        "reactants_smiles": "CN(C)C(=O)N(C)C.O=C(Cl)C(=O)Cl",
        "reference_conditions": {"solvent": "C(Cl)(Cl)(Cl)Cl", "catalyst": None, "temperature_celsius": None, "yield_percent": None},
    },
    {
        "reaction_id": "ord-1537debbdfcd4c9fb2d97af472b6fc7c",
        "product_smiles": "ClCC1CCN(Cc2ccccc2)C1",
        "reactants_smiles": "O=S(Cl)Cl.OCC1CCN(Cc2ccccc2)C1",
        "reference_conditions": {"solvent": "C(Cl)(Cl)Cl", "catalyst": None, "temperature_celsius": None, "yield_percent": None},
    },
    {
        "reaction_id": "ord-97da413a676c4eefa7c6d0c7176edaf1",
        "product_smiles": "CCCCOC(=O)COC(C)(C)C",
        "reactants_smiles": "C=C(C)C.CCCCOC(=O)CO.O=S(=O)(O)O",
        "reference_conditions": {"solvent": "C(Cl)Cl", "catalyst": None, "temperature_celsius": None, "yield_percent": None},
    },
    {
        "reaction_id": "ord-03bdefbab62e498087917c300d3854cf",
        "product_smiles": "COC1OC(=O)c2c(O)cccc21",
        "reactants_smiles": "CO.O=C1OC(O)c2cccc(O)c21",
        "reference_conditions": {"solvent": None, "catalyst": "S(O)(O)(=O)=O", "temperature_celsius": None, "yield_percent": None},
    },
]


def locate_checkpoint() -> str:
    configs = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
    assert configs, f"no config.json under /kaggle/input; contents: {os.listdir('/kaggle/input')}"
    print("candidate checkpoints:", configs)
    # /kaggle/input is read-only and the tie-weight fix below may rewrite config.json.
    source = os.path.dirname(configs[0])
    target = "/kaggle/working/model2"
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def fix_tie_word_embeddings(model_dir: str) -> None:
    """Auto-detect a config/weights mismatch on `tie_word_embeddings` (RESULTS.md).

    A checkpoint whose config claims tied embeddings while `lm_head.weight` was saved
    as a genuinely distinct tensor loads without error but generates garbage.
    """
    from safetensors import safe_open

    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(weights_path):
        return

    config = json.load(open(config_path, encoding="utf-8"))
    with safe_open(weights_path, framework="pt") as handle:
        keys = set(handle.keys())
        if "lm_head.weight" not in keys or "shared.weight" not in keys:
            return
        distinct = not torch.equal(handle.get_tensor("lm_head.weight"), handle.get_tensor("shared.weight"))

    if distinct and config.get("tie_word_embeddings", True):
        config["tie_word_embeddings"] = False
        json.dump(config, open(config_path, "w", encoding="utf-8"), indent=2)
        print("patched tie_word_embeddings -> False")


def format_input(row: dict) -> str:
    return f"predict conditions: PRODUCT: {row['product_smiles']} REACTANTS: {row['reactants_smiles']}"


def decode_and_parse(raw: str):
    for candidate in (raw, "{" + raw + "}"):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def main() -> None:
    model_dir = locate_checkpoint()
    fix_tie_word_embeddings(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).eval()

    prompts = [format_input(row) for row in RECORDS]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_BEAMS,
            max_length=MAX_TARGET_LENGTH,
        )
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    out = []
    for index, row in enumerate(RECORDS):
        raw_candidates = decoded[index * NUM_BEAMS : (index + 1) * NUM_BEAMS]
        seen, deduped = set(), []
        for candidate in raw_candidates:
            if candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        record = dict(row)
        record["predicted_conditions"] = [decode_and_parse(c) for c in deduped]
        record["predicted_conditions_raw"] = deduped
        out.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))

    with open("/kaggle/working/demo_cascade_out.json", "w", encoding="utf-8") as handle:
        json.dump({"num_beams": NUM_BEAMS, "records": out}, handle, ensure_ascii=False, indent=2)
    print("wrote /kaggle/working/demo_cascade_out.json")


main()
