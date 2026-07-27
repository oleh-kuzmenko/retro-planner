#!/usr/bin/env python3
"""Fine-tune a light T5 (t5-small/t5-base) to predict reaction conditions.

This is Model 2 of the diploma's own two-stage system: a plain, general-domain
T5 checkpoint (not chemically pretrained -- unlike Model 1's ReactionT5),
fully fine-tuned (no LoRA needed, the model is already small) on
`data/v2_ord_train/conditions_{train,val,test}.jsonl` (freshly sampled from
ORD, leak-checked against `data/v2_ord_eval_targets.json`, see
`build_train_data_ord.py`) to predict solvent/catalyst/temperature/yield from
a product + reactants pair.

Input format:  "predict conditions: PRODUCT: <product_smiles> REACTANTS: <reactants_smiles>"
Target format: a compact JSON string, e.g.
    {"solvent": "...", "catalyst": "...", "temperature_celsius": "...", "yield_percent": "..."}
Missing fields in the source row are written as "not specified", matching
this project's earlier condition-model evaluation convention (json validity,
has_solvent_or_reagents, etc. -- see `evaluate_conditions_model.py`).

Same Google Colab T4 (~3h/day) design as `train_reactant_model_ord.py`:
--output-dir must be Google-Drive-mounted, and re-running the same command
auto-resumes from the latest checkpoint; --time-budget-minutes force-saves
and stops cleanly before Colab's own timeout.

Example (every day, same command):
    python scripts/train_conditions_model.py \\
        --output-dir /content/drive/MyDrive/retro-planner-checkpoints/model2_conditions \\
        --time-budget-minutes 165
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

LOGGER_NAME = "retro_eval.train_conditions_model"

CONDITION_FIELDS = ("solvent", "catalyst", "temperature_celsius", "yield_percent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="t5-small", help="Plain (non-chemical) T5 checkpoint, e.g. t5-small or t5-base.")
    parser.add_argument("--train-file", type=Path, default=Path("data/v2_ord_train/conditions_train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/v2_ord_train/conditions_val.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True, help="Google-Drive-mounted path, not local disk.")
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-budget-minutes", type=float, default=None)
    return parser.parse_args()


def format_input(row: dict) -> str:
    return f"predict conditions: PRODUCT: {row['product_smiles']} REACTANTS: {row['reactants_smiles']}"


def format_target(row: dict) -> str:
    payload = {field: row.get(field) if row.get(field) not in (None, "") else "not specified" for field in CONDITION_FIELDS}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class TimeBudgetCallback:
    def __init__(self, budget_minutes: float | None):
        from transformers import TrainerCallback

        self._base = TrainerCallback
        self.budget_seconds = budget_minutes * 60 if budget_minutes else None
        self.start = time.monotonic()

    def build(self):
        budget_seconds = self.budget_seconds
        start = self.start

        class _Callback(self._base):
            def on_step_end(self, args, state, control, **kwargs):
                if budget_seconds is not None and (time.monotonic() - start) >= budget_seconds:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        return _Callback()


def main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(LOGGER_NAME)

    args = parse_args()

    from datasets import load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(args.seed)

    logger.info("Loading base checkpoint: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    data_files = {"train": str(args.train_file), "validation": str(args.val_file)}
    raw = load_dataset("json", data_files=data_files)
    logger.info("Train examples: %d | Validation examples: %d", len(raw["train"]), len(raw["validation"]))

    def preprocess(examples):
        inputs = [
            format_input({"product_smiles": p, "reactants_smiles": r})
            for p, r in zip(examples["product_smiles"], examples["reactants_smiles"])
        ]
        targets = [
            format_target(dict(zip(examples.keys(), values)))
            for values in zip(*examples.values())
        ]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        predict_with_generate=False,
        report_to=[],
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.add_callback(TimeBudgetCallback(args.time_budget_minutes).build())

    last_checkpoint = get_last_checkpoint(str(args.output_dir)) if args.output_dir.exists() else None
    if last_checkpoint:
        logger.info("Found existing checkpoint, resuming from: %s", last_checkpoint)
    else:
        logger.info("No existing checkpoint found in %s; starting fresh.", args.output_dir)

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    logger.info("Training finished (or paused at time budget). Latest state saved under %s", args.output_dir)


if __name__ == "__main__":
    main()
