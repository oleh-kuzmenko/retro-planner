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
Target format: selected by `--target-format`; which fields it carries by `--condition-fields`.

- `json` (default, what the t5-small runs in RESULTS.md used):
    {"solvent": "...", "catalyst": "...", "temperature_celsius": "...", "yield_percent": "..."}
  Missing fields are written as "not specified", matching this project's earlier
  condition-model evaluation convention (json validity, has_solvent_or_reagents,
  etc. -- see `evaluate_conditions_model.py`).
- `compact`: the values separated by `|`, missing ones written as `?`:
    CO|[Pd]|25.0|85.0
  Field names and braces carry no information the field order does not already
  carry, but they dominate the target: the JSON form of that example is 53
  tokens under t5-small's vocabulary against 13 for the compact one. The gap is
  decisive for a chemically pretrained base, whose vocabulary holds SMILES and
  not English -- `sagawa/CompoundT5` tokenizes the same JSON target into 74
  tokens of which 48 are `<unk>`, against 16 tokens and 9 `<unk>` for the
  compact form, and those 9 are repaired by `ensure_full_char_coverage` below.
  That `<unk>` corruption, not the base itself, is what made an earlier
  ReactionT5-based conditions run emit 0% parseable output.

The chosen format is recorded in `{output_dir}/final/conditions_format.json` so
the evaluation scripts parse generations the same way they were written.

Same Google Colab T4 (~3h/day) design as `train_reactant_model_ord.py`:
--output-dir must be Google-Drive-mounted; Trainer itself checkpoints and
rotates on local disk (--local-work-dir), and a callback syncs the latest
checkpoint out to `{output_dir}/latest_checkpoint` after every save (always
overwriting the same filenames in place, never deleting on Drive -- see
train_reactant_model_ord.py's docstring for why: Drive's FUSE mount moves
deletes to Trash instead of freeing them, which broke naive rotation
straight on Drive). --time-budget-minutes force-saves and stops cleanly
before Colab's own timeout.

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

from retro_eval.tokenizer_coverage import ensure_full_char_coverage

LOGGER_NAME = "retro_eval.train_conditions_model"

# `reagents` first: it is the longest field and the one the decoder is least able to guess
# from a prefix, so it gets the full attention budget rather than what is left after three
# other fields. It carries the stoichiometric species -- bases, coupling agents, halide
# sources -- that ORD writes on the reactant side and that neither model predicted before
# (`scripts/build_conditions_roles.py`).
CONDITION_FIELDS = ("reagents", "solvent", "catalyst", "temperature_celsius", "yield_percent")

TARGET_FORMATS = ("json", "compact")
# `|` never occurs in SMILES nor in ORD's condition strings (which join multiple
# components with ", "), so it separates the fields unambiguously.
COMPACT_SEPARATOR = "|"
COMPACT_MISSING = "?"
FORMAT_MARKER_FILE = "conditions_format.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="t5-small", help="Plain (non-chemical) T5 checkpoint, e.g. t5-small or t5-base.")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint or weights-only model folder to continue from -- e.g. a "
        "previous Colab session's {output_dir}/final, downloaded and re-uploaded to this "
        "session (possibly under a different Google account/Drive than last time). If the "
        "folder contains trainer_state.json (a full Trainer checkpoint), resumes exactly; "
        "otherwise it's loaded as the starting model for a fresh optimizer/step-count run. "
        "Takes priority over automatic local/Drive latest_checkpoint detection.",
    )
    parser.add_argument("--train-file", type=Path, default=Path("data/v2_ord_train/conditions_train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/v2_ord_train/conditions_val.jsonl"))
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
        default=Path("/content/local_model2_work"),
        help="Local (non-Drive) scratch directory where Trainer actually checkpoints "
        "and rotates old checkpoints. Wiped between Colab sessions -- that's fine, "
        "--output-dir/latest_checkpoint is what survives across sessions.",
    )
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument(
        "--target-format",
        choices=TARGET_FORMATS,
        default="json",
        help="How the four condition fields are serialized as the generation target. "
        "`json` reproduces the t5-small runs already in RESULTS.md; `compact` drops the "
        "field names and braces, which a SMILES-only vocabulary cannot represent.",
    )
    parser.add_argument(
        "--condition-fields",
        default=",".join(CONDITION_FIELDS),
        help="Comma-separated subset of the condition fields to predict, in any order "
        "(they are always emitted in CONDITION_FIELDS order). Dropping a field removes it "
        "from the target entirely rather than writing it as missing -- used to drop "
        "`yield_percent`, which is not a function of the reactant set: it was populated in "
        "60.7%% of training rows, yet the four-field model emitted a number for only 61 of "
        "5687 test records and abstained on 5234, i.e. it learned that abstaining beats any "
        "guess. The evaluator reads the field list back from the format marker.",
    )
    parser.add_argument(
        "--always-fields",
        default="",
        help="Comma-separated fields the model must never abstain on (e.g. "
        "'solvent,temperature_celsius'). Such a field is never written as `?`: a row that "
        "does not record it produces a shorter target without that slot, and the input "
        "carries a schema code saying which layout is expected -- the task-prefix scheme of "
        "T5 (Raffel et al., 2020) as used for chemistry in T5Chem (Lu and Zhang, 2022). At "
        "inference the full code is always requested, so the model emits every mandatory "
        "field. See `row_fields` for why `?` and row-dropping both fail here.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
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
        "optim=adafactor to keep each checkpoint's disk footprint down.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ignore-data-skip",
        action="store_true",
        help="On resume, start from the current position in the data stream instead of "
        "replaying the batches already consumed. Trainer's default replays them, which costs "
        "real minutes once a run is thousands of steps in -- the batches are still collated, "
        "only the forward pass is skipped. The price is that the resumed epoch sees a "
        "different sample order than it would have; with a shuffled sampler that is a "
        "reordering, not a change of the data.",
    )
    parser.add_argument("--time-budget-minutes", type=float, default=None)
    return parser.parse_args()


# One letter per field that the model must never be allowed to skip. The letters are joined
# in CONDITION_FIELDS order, so a row whose solvent and temperature are both recorded gets
# the schema code "ST" and one with only a solvent gets "S".
SCHEMA_CODES = {"reagents": "R", "solvent": "S", "catalyst": "C", "temperature_celsius": "T", "yield_percent": "Y"}
EMPTY_SCHEMA = "0"


def schema_code(row: dict, always_fields: tuple[str, ...]) -> str:
    """Which of the mandatory fields this row can actually supervise."""
    known = [SCHEMA_CODES[field] for field in always_fields if row.get(field) not in (None, "")]
    return "".join(known) or EMPTY_SCHEMA


def full_schema_code(always_fields: tuple[str, ...]) -> str:
    """The code used at inference: every mandatory field present, so the model must emit them."""
    return "".join(SCHEMA_CODES[field] for field in always_fields) or EMPTY_SCHEMA


def row_fields(row: dict, fields: tuple[str, ...], always_fields: tuple[str, ...]) -> tuple[str, ...]:
    """The target layout for one row: a mandatory field the record lacks is left out entirely.

    A field can be missing from ORD's record for two different reasons, and the difference
    decides how it must be written down. A catalyst is often genuinely absent, so `?` is the
    true answer and the model has to learn to produce it. A temperature is never absent from
    the reaction -- ORD simply did not log it -- and a solvent almost never is. Writing `?`
    there teaches the model to abstain on a field that always has an answer, and since `?` is
    the majority value for temperature (74% of rows), abstention becomes the loss-minimizing
    answer: the earlier B3 run stayed silent on 86.5% of test records. Dropping such rows
    instead costs the other fields their training data -- the earlier T2 run, trained only on
    rows carrying a temperature, scored 9.6% on the catalyst outside that subset against
    B3's 29.3%.

    So the row keeps all of its supervision, and the fields it cannot supervise simply do not
    appear in its target. The input's schema code says which layout to produce.
    """
    return tuple(field for field in fields if field not in always_fields or row.get(field) not in (None, ""))


def format_input(row: dict, schema: str = "") -> str:
    marker = f"[{schema}] " if schema else ""
    return f"predict conditions: {marker}PRODUCT: {row['product_smiles']} REACTANTS: {row['reactants_smiles']}"


def parse_condition_fields(spec: str) -> tuple[str, ...]:
    """Resolve a comma-separated `--condition-fields` value, keeping CONDITION_FIELDS order."""
    requested = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in requested if name not in CONDITION_FIELDS]
    if unknown:
        raise ValueError(f"Unknown condition field(s) {unknown}; choose from {list(CONDITION_FIELDS)}")
    if not requested:
        raise ValueError("--condition-fields must name at least one field")
    return tuple(field for field in CONDITION_FIELDS if field in requested)


def format_target(row: dict, target_format: str = "json", fields: tuple[str, ...] = CONDITION_FIELDS) -> str:
    if target_format == "compact":
        values = [
            str(row.get(field)) if row.get(field) not in (None, "") else COMPACT_MISSING
            for field in fields
        ]
        return COMPACT_SEPARATOR.join(values)
    payload = {field: row.get(field) if row.get(field) not in (None, "") else "not specified" for field in fields}
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
                if budget_seconds is None:
                    return control
                # Every rank runs its own clock, and torchrun starts them a second or two
                # apart, so "elapsed >= budget" turns true at different steps on different
                # ranks. Trainer does not synchronise `control`, so the rank that leaves the
                # loop first never issues the next collective and the other one waits on an
                # ALLREDUCE until NCCL's 30-minute watchdog kills the job -- which is exactly
                # how the first resumed M2 run died, 31 minutes after its last checkpoint.
                # The decision is therefore made collectively, on a step boundary all ranks
                # share.
                if state.global_step % 50:
                    return control
                expired = (time.monotonic() - start) >= budget_seconds

                import torch
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    flag = torch.tensor([1.0 if expired else 0.0], device=args.device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    expired = bool(flag.item())

                if expired:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        return _Callback()


def materialize_writable_checkpoint(checkpoint: Path, local_work_dir: Path, logger) -> Path:
    """Return a writable copy of `checkpoint`, or the original if it is already writable.

    Resuming rewrites `best_model_checkpoint` inside `trainer_state.json`, so the checkpoint
    directory has to be writable. It is not when it arrives as another kernel's output under
    Kaggle's read-only `/kaggle/input`, and the run dies several minutes in with
    `OSError: [Errno 30] Read-only file system`.
    """
    import os
    import shutil

    if os.access(checkpoint, os.W_OK):
        return checkpoint
    destination = local_work_dir / f"resume_checkpoint_rank{os.environ.get('RANK', '0')}"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Resume checkpoint %s is read-only; copying it to %s.", checkpoint, destination)
    shutil.copytree(checkpoint, destination)
    # copytree preserves permission bits, so a source that is read-only by mode rather than
    # by mount would produce an equally unwritable copy.
    for path in (destination, *destination.rglob("*")):
        path.chmod(path.stat().st_mode | 0o200 | (0o100 if path.is_dir() else 0))
    return destination


def is_main_process() -> bool:
    """Whether this process should do exclusive work (Drive sync, final save) under multi-GPU DDP.

    See `train_reactant_model_ord.py`'s `is_main_process` for why `RANK` is checked
    directly rather than trusting `Trainer.is_world_process_zero()`: on at least one
    environment both processes reported themselves as world-process-zero and raced to
    write the same files. Defaults to "0" (true) for a plain single-process `python`
    launch (e.g. Colab), where `RANK` is never set.
    """
    import os

    return os.environ.get("RANK", "0") == "0"


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
                if not is_main_process():
                    # Under multi-GPU DDP (e.g. `torchrun --nproc_per_node=2`), every
                    # rank runs this callback; only rank 0 should touch the Drive copy,
                    # otherwise concurrent shutil.copytree calls race on the same files.
                    return control
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
    for noisy_logger in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
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

    set_seed(args.seed)

    model_load_path = args.base_model
    explicit_resume_checkpoint = None
    if args.resume_from_checkpoint is not None:
        if is_valid_checkpoint_dir(args.resume_from_checkpoint):
            explicit_resume_checkpoint = str(args.resume_from_checkpoint)
            logger.info(
                "--resume-from-checkpoint %s is a full Trainer checkpoint; resuming exactly "
                "(optimizer state, step count preserved).",
                explicit_resume_checkpoint,
            )
        else:
            model_load_path = str(args.resume_from_checkpoint)
            logger.info(
                "--resume-from-checkpoint %s has no trainer_state.json (a weights-only folder, "
                "e.g. 'final'); loading it as the starting model for a fresh optimizer/step-count run.",
                model_load_path,
            )

    logger.info("Loading base checkpoint: %s", model_load_path)
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_load_path)

    data_files = {"train": str(args.train_file), "validation": str(args.val_file)}
    raw = load_dataset("json", data_files=data_files)
    logger.info("Train examples: %d | Validation examples: %d", len(raw["train"]), len(raw["validation"]))

    fields = parse_condition_fields(args.condition_fields)
    logger.info("Predicting %d condition field(s): %s", len(fields), ", ".join(fields))

    always_fields = parse_condition_fields(args.always_fields) if args.always_fields.strip() else ()
    unlisted = [field for field in always_fields if field not in fields]
    if unlisted:
        raise ValueError(f"--always-fields names {unlisted}, which --condition-fields does not predict")
    if always_fields:
        counts: dict[str, int] = {}
        for row in raw["train"]:
            counts[schema_code(row, always_fields)] = counts.get(schema_code(row, always_fields), 0) + 1
        logger.info(
            "Mandatory field(s) %s | schema codes in train: %s | inference code: %s",
            ", ".join(always_fields),
            ", ".join(f"{code} {count}" for code, count in sorted(counts.items(), key=lambda kv: -kv[1])),
            full_schema_code(always_fields),
        )

    all_rows = [row for split in ("train", "validation") for row in raw[split]]
    all_texts = [format_input(row, schema_code(row, always_fields)) for row in all_rows] + [
        format_target(row, args.target_format, row_fields(row, fields, always_fields)) for row in all_rows
    ]
    ensure_full_char_coverage(tokenizer, model, all_texts, logger)

    def preprocess(examples):
        rows = [dict(zip(examples.keys(), values)) for values in zip(*examples.values())]
        inputs = [format_input(row, schema_code(row, always_fields)) for row in rows]
        targets = [
            format_target(row, args.target_format, row_fields(row, fields, always_fields))
            for row in rows
        ]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.local_work_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.local_work_dir),
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
        disable_tqdm=not args.enable_tqdm,
        log_level="warning",
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adafactor",
        report_to=[],
        seed=args.seed,
        ignore_data_skip=args.ignore_data_skip,
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

    if args.resume_from_checkpoint is not None:
        last_checkpoint = explicit_resume_checkpoint
        if explicit_resume_checkpoint:
            # A checkpoint handed over from another Kaggle kernel arrives under
            # /kaggle/input, which is mounted read-only, and the repoint below writes to
            # trainer_state.json. Copy it somewhere writable first. The copy is per-rank so
            # that two DDP processes never write the same files; the source is identical, so
            # so are the copies.
            explicit_resume_checkpoint = str(
                materialize_writable_checkpoint(Path(explicit_resume_checkpoint), args.local_work_dir, logger)
            )
            last_checkpoint = explicit_resume_checkpoint
            state_path = Path(explicit_resume_checkpoint) / "trainer_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("best_model_checkpoint") and not Path(state["best_model_checkpoint"]).exists():
                logger.info(
                    "trainer_state.json's best_model_checkpoint (%s) doesn't exist in this "
                    "session/environment; repointing it to %s.",
                    state["best_model_checkpoint"],
                    explicit_resume_checkpoint,
                )
                state["best_model_checkpoint"] = explicit_resume_checkpoint
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        else:
            logger.info(
                "--resume-from-checkpoint weights already loaded as the starting model; "
                "beginning a fresh optimizer/step-count run from them."
            )
    else:
        from transformers.trainer_utils import get_last_checkpoint

        last_checkpoint = get_last_checkpoint(str(args.local_work_dir))
        if last_checkpoint:
            logger.info("Found local checkpoint (same-session restart), resuming from: %s", last_checkpoint)
        elif is_valid_checkpoint_dir(drive_resume_dir):
            last_checkpoint = str(drive_resume_dir)
            logger.info("Found Drive checkpoint from a previous session, resuming from: %s", last_checkpoint)
            # See train_reactant_model_ord.py for why this repoint is needed: the
            # previous session's local_work_dir path in best_model_checkpoint no
            # longer exists once local disk is wiped between sessions.
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
            last_checkpoint = None
            logger.info(
                "No valid checkpoint in %s (local) or %s (Drive); starting fresh.",
                args.local_work_dir,
                drive_resume_dir,
            )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    if is_main_process():
        # Same reasoning as DriveSyncCallback: under multi-GPU DDP, every rank would
        # otherwise race to write the same "final" folder.
        trainer.save_model(str(args.output_dir / "final"))
        tokenizer.save_pretrained(str(args.output_dir / "final"))
        # Generations are only parseable if the reader knows how they were written.
        (args.output_dir / "final" / FORMAT_MARKER_FILE).write_text(
            json.dumps(
                {
                    "target_format": args.target_format,
                    "fields": list(fields),
                    "always_fields": list(always_fields),
                }
            ),
            encoding="utf-8",
        )
        logger.info("Training finished (or paused at time budget). Latest state saved under %s", args.output_dir)


if __name__ == "__main__":
    main()
