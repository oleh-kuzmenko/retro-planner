#!/usr/bin/env python3
"""Train Model 2 as a classifier: the t5-small encoder plus one head per condition field.

`train_conditions_model.py` writes the conditions as text and reads them back with beam
search. That spends the candidate list badly -- five beams of the generative run carry only
2.84 distinct solvents and 1.18 distinct catalysts, because beams diverge on how a formula is
spelled long before they diverge on which substance is named (RESULTS.md, "E0"). A softmax
cannot do that: its five highest classes are five different answers by construction.

The reformulation is possible because the answer is a choice, not a structure. Solvent and
catalyst are drawn from a small set of laboratory reagents -- the 1,000 most frequent
component sets of the training corpus cover 95.1% and 97.1% of the test references -- and
temperatures are written as round numbers, so 200 distinct values cover 99.4% of them. A class
is the whole component *set*, which is exactly what `exact_match` compares, so mixtures need no
special handling: "DCM + water" is its own class and the order it was written in is gone before
training starts.

Temperature stays a number. Its classes are the values themselves (0, 80, 100, -78 ...), not
ranges: a range would leave `within_tol` (+-10 C) uncomputable and hand the chemist an interval
where the protocol needs a figure.

Incomplete annotation is handled by the loss, not by a prefix scheme:

- solvent, catalyst: an explicit "not specified" class. For the catalyst that is a real answer
  (the record scale scores correct silence), and it costs one of the five slots at most, so the
  strict accuracy over the records that *do* name one is untouched.
- temperature: rows without one are masked out of the loss entirely. Absence there means the
  protocol did not record the number, never that the reaction has no temperature -- training on
  it is what taught the generative runs to answer with the missing marker.

Example:
    torchrun --nproc_per_node=2 scripts/train_conditions_classifier.py \\
        --train-file data/v2_ord_roles_full/conditions_train.jsonl \\
        --val-file data/v2_ord_roles_full/conditions_val.jsonl \\
        --output-dir /content/drive/MyDrive/retro-planner-checkpoints/e1_classifier
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_conditions_model import normalize_components, to_number
from train_conditions_model import (
    CONDITION_FIELDS,
    REACTANTS_FIELDS,
    DriveSyncCallback,
    TimeBudgetCallback,
    format_input,
    parse_condition_fields,
)

logger = logging.getLogger("train_conditions_classifier")

ABSENT = "not specified"
META_FILE = "classifier_meta.json"
HEADS_FILE = "heads.pt"
NUMERIC_FIELDS = ("temperature_celsius", "yield_percent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-model", default="t5-small")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-work-dir", type=Path, default=Path("/tmp/local_classifier"))
    parser.add_argument("--condition-fields", default="solvent,catalyst,temperature_celsius")
    parser.add_argument("--reactants-field", default=REACTANTS_FIELDS[0], choices=REACTANTS_FIELDS)
    parser.add_argument(
        "--num-classes",
        default="solvent=1000,catalyst=1000,temperature_celsius=200",
        help="Per-field class budget, counted off the training corpus by frequency.",
    )
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=200)
    parser.add_argument("--time-budget-minutes", type=float, default=None)
    parser.add_argument("--group-by-length", action="store_true")
    parser.add_argument("--enable-tqdm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_num_classes(spec: str, fields: tuple[str, ...]) -> dict[str, int]:
    budget = {}
    for item in spec.split(","):
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in CONDITION_FIELDS:
            raise ValueError(f"--num-classes names unknown field {name!r}")
        budget[name] = int(value)
    missing = [field for field in fields if field not in budget]
    if missing:
        raise ValueError(f"--num-classes has no budget for {missing}")
    return {field: budget[field] for field in fields}


def class_key(field: str, value):
    """The identity the metric itself uses, so a class is what `exact_match` compares."""
    if field in NUMERIC_FIELDS:
        return to_number(value)
    key = normalize_components(value)
    return None if key is None else ".".join(sorted(key))


def read_rows(path: Path):
    """Stream a corpus file. The 486,330-row corpus is never held in memory as dicts: two
    processes each materializing it is what an OOM kill on a 2xT4 Kaggle node looks like."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def build_vocabularies(path: Path, fields: tuple[str, ...], budget: dict[str, int]) -> dict:
    """Most frequent values per field, plus the absent class where absence is an answer."""
    counters = {field: collections.Counter() for field in fields}
    absences = {field: 0 for field in fields}
    for row in read_rows(path):
        for field in fields:
            key = class_key(field, row.get(field))
            if key is None:
                absences[field] += 1
            else:
                counters[field][key] += 1

    vocabularies = {}
    for field in fields:
        counts = counters[field]
        absent = absences[field]
        classes = [key for key, _ in counts.most_common(budget[field])]
        covered = sum(counts[key] for key in classes)
        named = sum(counts.values())
        # Temperature is masked instead: absence there is an unrecorded number, not a fact
        # about the reaction, and a class for it is what taught the generative runs to abstain.
        has_absent = field not in NUMERIC_FIELDS
        if has_absent:
            classes = [ABSENT] + classes
        vocabularies[field] = {
            "classes": classes,
            "absent_class": 0 if has_absent else None,
            "train_rows_with_value": named,
            "train_rows_absent": absent,
            "coverage_of_named_rows": covered / named if named else 0.0,
        }
        logger.info(
            "%s: %d classes%s, covering %.1f%% of the %d rows that name one (%d absent)",
            field,
            len(classes),
            " incl. absent" if has_absent else " (absent rows masked)",
            100 * covered / named if named else 0.0,
            named,
            absent,
        )
    return vocabularies


def label_of(field: str, value, vocabulary: dict, index: dict) -> int:
    """-100 marks a row this field cannot learn from: absent-and-masked, or out of vocabulary."""
    key = class_key(field, value)
    if key is None:
        absent = vocabulary["absent_class"]
        return -100 if absent is None else absent
    return index.get(key, -100)


class ConditionClassifier(nn.Module):
    """Pooled encoder states, one linear head per field.

    Mean pooling over the unpadded tokens, not the first position: T5 has no [CLS] and its
    encoder was never trained to summarize a sequence into one slot.
    """

    # Read by Trainer when it reloads the best checkpoint; a plain module has no such
    # attribute and the lookup would raise instead of warning about the dropped alias.
    _keys_to_ignore_on_save = None

    def __init__(self, encoder, head_sizes: dict[str, int]):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.fields = tuple(head_sizes)
        self.heads = nn.ModuleDict(
            {field: nn.Linear(encoder.config.d_model, size) for field, size in head_sizes.items()}
        )

    def state_dict(self, *args, **kwargs):
        """Drop the embedding alias before anything tries to serialize it.

        T5 points `encoder.embed_tokens.weight` at `shared.weight`, one buffer under two
        names. `PreTrainedModel.save_pretrained` knows to drop the alias; safetensors, which
        the Trainer reaches for on a plain module, refuses to write it at all -- and the
        `save_safetensors` switch that would sidestep it does not exist in every transformers
        release this repo runs on. Dropping the duplicate name is version-independent: the
        tie is rebuilt on load, because both names still point at the one parameter.
        """
        state = super().state_dict(*args, **kwargs)
        for key in [name for name in state if name.endswith("encoder.encoder.embed_tokens.weight")]:
            state.pop(key)
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict=False, assign=assign)

    def forward(self, input_ids=None, attention_mask=None, **labels):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        logits = {field: self.heads[field](pooled) for field in self.fields}

        loss = None
        for field in self.fields:
            target = labels.get(f"labels_{field}")
            if target is None:
                continue
            # A batch can hold no usable row for a sparsely annotated field; cross_entropy
            # returns nan there, which would poison the whole step.
            if int((target != -100).sum()) == 0:
                continue
            field_loss = F.cross_entropy(logits[field], target, ignore_index=-100)
            loss = field_loss if loss is None else loss + field_loss
        if loss is None:
            loss = sum(head.weight.sum() for head in self.heads.values()) * 0.0

        return {"loss": loss, **{f"logits_{field}": value for field, value in logits.items()}}


def save_classifier(model, tokenizer, meta: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    core = model.module if hasattr(model, "module") else model
    core.encoder.save_pretrained(destination / "encoder")
    tokenizer.save_pretrained(destination / "encoder")
    torch.save(core.heads.state_dict(), destination / HEADS_FILE)
    (destination / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_classifier(source: Path):
    from transformers import AutoTokenizer, T5EncoderModel

    meta = json.loads((source / META_FILE).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(source / "encoder")
    encoder = T5EncoderModel.from_pretrained(source / "encoder")
    head_sizes = {field: len(meta["vocabularies"][field]["classes"]) for field in meta["fields"]}
    model = ConditionClassifier(encoder, head_sizes)
    model.heads.load_state_dict(torch.load(source / HEADS_FILE, map_location="cpu"))
    return model, tokenizer, meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        T5EncoderModel,
        Trainer,
        TrainingArguments,
    )
    from retro_eval.tokenizer_coverage import ensure_full_char_coverage

    fields = parse_condition_fields(args.condition_fields)
    budget = parse_num_classes(args.num_classes, fields)

    raw = load_dataset(
        "json", data_files={"train": str(args.train_file), "validation": str(args.val_file)}
    )
    logger.info("Train examples: %d | validation: %d", len(raw["train"]), len(raw["validation"]))
    logger.info("Reactant side of the input: %s", args.reactants_field)

    vocabularies = build_vocabularies(args.train_file, fields, budget)
    indexes = {
        field: {key: position for position, key in enumerate(vocabularies[field]["classes"])}
        for field in fields
    }

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    encoder = T5EncoderModel.from_pretrained(args.base_model)
    texts = (
        format_input(row, "", args.reactants_field)
        for path in (args.train_file, args.val_file)
        for row in read_rows(path)
    )
    ensure_full_char_coverage(tokenizer, encoder, texts, logger)

    model = ConditionClassifier(encoder, {field: len(vocabularies[field]["classes"]) for field in fields})

    def preprocess(examples):
        rows = [dict(zip(examples.keys(), values)) for values in zip(*examples.values())]
        encoded = tokenizer(
            [format_input(row, "", args.reactants_field) for row in rows],
            max_length=args.max_source_length,
            truncation=True,
        )
        for field in fields:
            encoded[f"labels_{field}"] = [
                label_of(field, row.get(field), vocabularies[field], indexes[field]) for row in rows
            ]
        return encoded

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    train_dataset, val_dataset = tokenized["train"], tokenized["validation"]
    for field in fields:
        labelled = vocabularies[field]["train_rows_with_value"]
        if vocabularies[field]["absent_class"] is not None:
            labelled += vocabularies[field]["train_rows_absent"]
        logger.info("%s: %d of %d training rows carry a label", field, labelled, len(train_dataset))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.local_work_dir.mkdir(parents=True, exist_ok=True)

    length_grouping = {}
    if args.group_by_length:
        import dataclasses

        if "group_by_length" in {f.name for f in dataclasses.fields(TrainingArguments)}:
            length_grouping["group_by_length"] = True

    training_args = TrainingArguments(
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
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adafactor",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        label_names=[f"labels_{field}" for field in fields],
        **length_grouping,
    )

    from transformers import DataCollatorWithPadding

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.add_callback(TimeBudgetCallback(args.time_budget_minutes).build())
    trainer.add_callback(DriveSyncCallback(args.local_work_dir, args.output_dir / "latest_checkpoint", logger).build())

    trainer.train()

    meta = {
        "base_model": args.base_model,
        "fields": list(fields),
        "reactants_field": args.reactants_field,
        "max_source_length": args.max_source_length,
        "vocabularies": vocabularies,
        "train_rows": len(raw["train"]),
    }
    if os.environ.get("RANK", "0") == "0":
        save_classifier(model, tokenizer, meta, args.output_dir / "final")
        logger.info("Saved to %s", args.output_dir / "final")


if __name__ == "__main__":
    main()
