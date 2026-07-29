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
  `/content/drive/MyDrive/retro-planner-checkpoints/model1_reactant`) --
  this is where the *cross-session* resume checkpoint and the final model
  end up, so they survive a Colab disconnect/session reset.
- **Trainer itself checkpoints and rotates on local Colab disk**
  (`--local-work-dir`, wiped between sessions), NOT directly on the
  Drive-mounted path. Earlier versions had Trainer's own `save_total_limit`
  rotation running straight on Drive, and Google Drive's FUSE mount moves
  deleted files to Trash instead of freeing them -- besides not reclaiming
  quota, this was observed to occasionally corrupt the rotation bookkeeping
  (partially-restored/empty checkpoint folders, resume picking the wrong
  one, or not resuming at all). Local disk has none of that: Trainer's
  rotation is fast, reliable, and ordinary POSIX semantics. After every
  save, a callback copies (overwrites in place, same filenames every time --
  never deletes anything on Drive) the just-written local checkpoint to
  `{output_dir}/latest_checkpoint`, which is the one and only thing this
  script ever reads back from Drive to resume.
- `--time-budget-minutes` stops training early (forcing an immediate
  checkpoint save + Drive sync) before Colab's own session timeout, instead
  of losing a partially-completed step.

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
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Google-Drive-mounted path. Only ever written to as: overwrite-in-place "
        "{output_dir}/latest_checkpoint (cross-session resume point) and one-time "
        "{output_dir}/final -- never rotated/deleted, so Drive's Trash-on-delete "
        "behavior never comes into play.",
    )
    parser.add_argument(
        "--local-work-dir",
        type=Path,
        default=Path("/content/local_model1_work"),
        help="Local (non-Drive) scratch directory where Trainer actually checkpoints "
        "and rotates old checkpoints (fast, reliable, ordinary filesystem semantics). "
        "Wiped between Colab sessions -- that's fine, --output-dir/latest_checkpoint "
        "is what survives across sessions.",
    )
    parser.add_argument("--max-source-length", type=int, default=150)
    parser.add_argument("--max-target-length", type=int, default=150)
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=2.0,
        help="Lowered from 3.0: v3 showed eval_loss bottoming out and then climbing "
        "monotonically from ~7% into epoch 1 already, i.e. well under one epoch -- "
        "further epochs mostly burn Colab time past the point load_best_model_at_end "
        "will roll back to anyway.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Lowered from 5e-5 (already lowered once from an initial 3e-4 -- see "
        "diploma methodology notes): v3 at 5e-5 still overfit within well under one "
        "epoch (eval_loss rising monotonically from the first measurement), so "
        "trying a gentler rate to see whether the useful-improvement window widens.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Was unset (0.0) in v2/v3; a small decay is standard practice and a cheap "
        "extra guard against the fast overfitting v3 showed.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=0.5,
        help="Probability of showing a randomized (vs. canonical) SMILES rendering for "
        "each training example on each access. v3 always randomized (prob=1.0), which "
        "may have widened the train/validation distribution gap (validation always "
        "stays canonical) and contributed to the fast overfitting observed -- 0.5 keeps "
        "most of the augmentation benefit while regularly showing the same canonical "
        "form validation uses.",
    )
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=200,
        help="Kept high (was 25) -- Colab's browser tab can hang after a couple hours if "
        "the notebook cell's output/DOM grows too large from frequent log lines and "
        "per-step tqdm bar updates. The per-step progress bar itself is disabled "
        "unconditionally (see --enable-tqdm) in favor of this periodic printout.",
    )
    parser.add_argument(
        "--enable-tqdm",
        action="store_true",
        help="Re-enable the per-step progress bar (off by default -- see --logging-steps).",
    )
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
        "--no-augment",
        action="store_true",
        help="Disable SMILES-randomization data augmentation (on by default). Literature "
        "(Tetko et al. 2020; RSGPT, Nat. Commun. 2025) reports roughly +10-14 points "
        "absolute top-1 accuracy from training on randomized, non-canonical SMILES "
        "instead of memorizing one canonical string per molecule. Applied online (a "
        "fresh random rendering each epoch) to the train split only -- validation stays "
        "canonical/deterministic so eval_loss is comparable across checkpoints.",
    )
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


def is_valid_checkpoint_dir(path: Path) -> bool:
    """Whether `path` has the minimum files needed to resume from it."""
    if not path.is_dir():
        return False
    has_weights = any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
    has_trainer_state = (path / "trainer_state.json").exists()
    return has_weights and has_trainer_state


class DriveSyncCallback:
    """Copies the just-written local checkpoint out to a single fixed Drive folder.

    HF checkpoint directories always use the same fixed filenames
    (model.safetensors, optimizer.pt, trainer_state.json, ...) regardless of
    step number, so copying into a fixed-name destination overwrites those
    files in place -- a plain write, never a delete -- which sidesteps
    Google Drive's Trash-on-delete behavior entirely. Only one checkpoint
    ever lives on Drive at a time; local disk (--local-work-dir) is where
    Trainer's own save_total_limit rotation actually happens.
    """

    def __init__(self, local_work_dir: Path, drive_resume_dir: Path, logger):
        from transformers import TrainerCallback

        self._base = TrainerCallback
        self.local_work_dir = local_work_dir
        self.drive_resume_dir = drive_resume_dir
        self.logger = logger

    def build(self):
        import shutil

        local_work_dir = self.local_work_dir
        drive_resume_dir = self.drive_resume_dir
        logger = self.logger

        class _Callback(self._base):
            def on_save(self, args, state, control, **kwargs):
                local_checkpoint = local_work_dir / f"checkpoint-{state.global_step}"
                if not is_valid_checkpoint_dir(local_checkpoint):
                    logger.warning("Expected local checkpoint %s not found; skipping Drive sync.", local_checkpoint)
                    return control
                drive_resume_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(local_checkpoint, drive_resume_dir, dirs_exist_ok=True)
                logger.info("Synced checkpoint (step %d) to Drive: %s", state.global_step, drive_resume_dir)
                return control

        return _Callback()


def main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # huggingface_hub/urllib3 log an INFO line per HTTP request (model/tokenizer
    # download HEAD/GET calls) -- harmless but adds noise; only training's own
    # logger and Trainer's own periodic logging_steps output should be visible.
    for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
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

    set_seed(args.seed)

    logger.info("Loading base checkpoint: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    # sagawa/ReactionT5v2-* checkpoints ship config.tie_word_embeddings=True but
    # lm_head.weight actually differs from shared.weight in the checkpoint, so
    # transformers refuses to tie them at load time and prints a warning.
    # Matching the config to that reality silences it. (Separately,
    # encoder.embed_tokens/decoder.embed_tokens always alias the same `shared`
    # module by T5's architecture -- save/reload logs a "missing keys" note for
    # those two, which is expected and harmless; verified after a smoke run
    # that the reloaded embeddings are non-random and reflect training, not a
    # reinit.)
    model.config.tie_word_embeddings = False

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

    validation_tokenized = raw["validation"].map(
        preprocess, batched=True, remove_columns=raw["validation"].column_names
    )

    if args.no_augment:
        train_tokenized = raw["train"].map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    else:
        import numpy as np

        from retro_eval.chemistry import canonicalize_smiles, randomize_smiles

        rng = np.random.default_rng(args.seed)

        def augment_one(smiles: str) -> str:
            if rng.random() < args.augment_prob:
                return randomize_smiles(smiles, rng) or smiles
            return canonicalize_smiles(smiles) or smiles

        def preprocess_augmented(examples):
            product_smiles = [augment_one(p) for p in examples["product_smiles"]]
            reactants_smiles = [augment_one(r) for r in examples["reactants_smiles"]]
            model_inputs = tokenizer(product_smiles, max_length=args.max_source_length, truncation=True)
            labels = tokenizer(text_target=reactants_smiles, max_length=args.max_target_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        logger.info(
            "SMILES augmentation ON (prob=%.2f): train split re-randomized on every access (online, per-epoch).",
            args.augment_prob,
        )
        train_raw = raw["train"].remove_columns(
            [c for c in raw["train"].column_names if c not in ("product_smiles", "reactants_smiles")]
        )
        train_raw.set_transform(preprocess_augmented)
        train_tokenized = train_raw

    tokenized = {"train": train_tokenized, "validation": validation_tokenized}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.local_work_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.local_work_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        disable_tqdm=not args.enable_tqdm,
        log_level="warning",
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adafactor",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.add_callback(TimeBudgetCallback(args.time_budget_minutes).build())
    drive_resume_dir = args.output_dir / "latest_checkpoint"
    trainer.add_callback(DriveSyncCallback(args.local_work_dir, drive_resume_dir, logger).build())

    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = get_last_checkpoint(str(args.local_work_dir))
    if last_checkpoint:
        logger.info("Found local checkpoint (same-session restart), resuming from: %s", last_checkpoint)
    elif is_valid_checkpoint_dir(drive_resume_dir):
        last_checkpoint = str(drive_resume_dir)
        logger.info("Found Drive checkpoint from a previous session, resuming from: %s", last_checkpoint)
        # trainer_state.json's best_model_checkpoint points at the *previous*
        # session's local_work_dir path, which no longer exists (local disk is
        # wiped between sessions). If this session never finds a better eval_loss,
        # load_best_model_at_end would try to load that stale path at the very
        # end and crash. Since drive_resume_dir's weights are exactly what that
        # stale path would have pointed to anyway, repointing it here is safe.
        import json

        state_path = drive_resume_dir / "trainer_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("best_model_checkpoint") and not Path(state["best_model_checkpoint"]).exists():
            logger.info(
                "trainer_state.json's best_model_checkpoint (%s) is from a previous "
                "session's local disk; repointing it to %s.",
                state["best_model_checkpoint"],
                drive_resume_dir,
            )
            state["best_model_checkpoint"] = str(drive_resume_dir)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    else:
        logger.info(
            "No valid checkpoint in %s (local) or %s (Drive); starting fresh.",
            args.local_work_dir,
            drive_resume_dir,
        )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    logger.info("Training finished (or paused at time budget). Latest state saved under %s", args.output_dir)


if __name__ == "__main__":
    main()
