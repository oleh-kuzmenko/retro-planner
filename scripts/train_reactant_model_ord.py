#!/usr/bin/env python3
"""Fine-tune ReactionT5 (ORD-pretrained checkpoint) on the new ORD retrosynthesis train split.

This is Model 1 of the diploma's own two-stage system: starts from
`sagawa/ReactionT5v2-retrosynthesis` -- the checkpoint right after ORD
reaction-pretraining, *before* any USPTO-50k-specific fine-tuning (see
`research/README` / thesis Section 2 for why that starting point avoids
overwriting USPTO-specific patterns) -- and fine-tunes it on
`data/v2_ord_train/reactants_train.jsonl` (freshly sampled, leak-checked
against `data/v2_ord_eval_targets.json`, see `build_train_data_ord.py`).

Designed for Google Colab T4 sessions capped at ~3 hours/day:

- `--output-dir` should be a Google-Drive-mounted path (e.g.
  `/content/drive/MyDrive/retro-planner-checkpoints/model1_reactant`), NOT
  local Colab disk -- local `/content` is wiped between sessions.
- The script auto-resumes from the latest checkpoint already in
  `--output-dir`, via `transformers.trainer_utils.get_last_checkpoint`, so
  re-running the exact same command on a fresh Colab session continues
  training rather than restarting.
- `--time-budget-minutes` stops training early (forcing an immediate
  checkpoint save) before Colab's own session timeout, instead of losing a
  partially-completed step.

Example (every day, same command -- first run starts training, later runs resume):
    python scripts/train_reactant_model_ord.py \\
        --output-dir /content/drive/MyDrive/retro-planner-checkpoints/model1_reactant \\
        --time-budget-minutes 165

To also mix in the canonical USPTO-50K train split (guards against the
fine-tuned model regressing on USPTO-style products), add:
    --extra-train-file data/v2_uspto_train/reactants_train.jsonl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

LOGGER_NAME = "retro_eval.train_reactant_model_ord"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="sagawa/ReactionT5v2-retrosynthesis")
    parser.add_argument("--train-file", type=Path, default=Path("data/v2_ord_train/reactants_train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/v2_ord_train/reactants_val.jsonl"))
    parser.add_argument(
        "--extra-train-file",
        type=Path,
        default=None,
        help="Optional additional JSONL (e.g. data/v2_uspto_train/reactants_train.jsonl) mixed into training.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Google-Drive-mounted path, not local disk.")
    parser.add_argument("--max-source-length", type=int, default=150)
    parser.add_argument("--max-target-length", type=int, default=150)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Full fine-tune of an already-pretrained checkpoint; kept conservative "
        "(was 3e-4 in an earlier version, which measurably degraded ORD accuracy "
        "below the pre-fine-tune baseline -- see diploma methodology notes).",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Kept low for Google Drive's free-tier ~15GB quota; combined with "
        "optim=adafactor (much smaller optimizer state than AdamW) to keep each "
        "checkpoint's disk footprint down. load_best_model_at_end always protects "
        "the best checkpoint from rotation regardless of this limit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--time-budget-minutes",
        type=float,
        default=None,
        help="Soft wall-clock limit; a checkpoint is force-saved and training stops cleanly when reached.",
    )
    return parser.parse_args()


class TimeBudgetCallback:
    """HF TrainerCallback that force-stops (with a checkpoint) after a wall-clock budget."""

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

    from datasets import concatenate_datasets, load_dataset
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

    if args.extra_train_file is not None:
        logger.info("Mixing in extra train file: %s", args.extra_train_file)
        extra = load_dataset("json", data_files={"train": str(args.extra_train_file)})["train"]
        raw["train"] = concatenate_datasets([raw["train"], extra]).shuffle(seed=args.seed)

    logger.info("Train examples: %d | Validation examples: %d", len(raw["train"]), len(raw["validation"]))

    def preprocess(examples):
        model_inputs = tokenizer(
            examples["product_smiles"],
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["reactants_smiles"],
            max_length=args.max_target_length,
            truncation=True,
        )
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
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adafactor",
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
