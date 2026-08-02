# Agent Notes

Two-stage, fully self-trained retrosynthesis system (see `RESULTS.md` for all numbers).
Everything trains/evaluates on ORD; USPTO-50K is an independent generalization test only.
Inference: target SMILES → Model 1 → RDKit validity check → Model 2. No LLM API, no Qdrant.

## Data (leak-free by construction)

- `build_eval_targets_ord.py` → `data/v2_ord_eval_targets.json` (Model 1 held-out test, ~300).
- `build_train_data_ord.py` draws ONE eval-excluded, stratified ORD sample and derives both
  Model 1's reactant split and Model 2's condition train/val/test split from it — single
  source, zero leakage (verified: 0 `product_smiles` overlap across splits). `--pool-count`
  sizes it (60k/150k/300k pools were built to test data scaling).
- `build_root_aligned_data.py` rewrites reactant targets into root-aligned form (RXNMapper +
  RDKit) for Model 1 variant 6.
- If you touch any `data/v2_*` file, re-run the pairwise `product_smiles` leakage check
  before trusting new numbers.

## Model 1 — reactants (`train_reactant_model_ord.py`)

Fine-tunes `sagawa/ReactionT5v2-retrosynthesis` (ORD-pretrained, before Sagawa/Kojima's own
USPTO fine-tune) on the ORD reactant split. Proven config: `--no-augment --learning-rate
5e-5 --num-train-epochs 3`. Eval with `scripts/models/run_reactiont5_topk.py` (top-1/3/5
beam search); merge two checkpoints' candidates with `ensemble_topk.py` (the headline
82.3% top-5 core). Lessons learned:

- **DDP write races**: gate Drive-sync/final-save on `os.environ["RANK"]`, not
  `state.is_world_process_zero` (unreliable when transformers doesn't detect
  `ParallelMode.DISTRIBUTED`; both ranks then race to write the same files).
- SMILES augmentation and diverse beam search both *hurt* here; more data (150k/300k) did not
  beat the 57k config; round-trip forward-model reranking also lowered accuracy. All
  documented as negative results.

## Model 2 — conditions (`train_conditions_model.py`)

Fine-tunes plain `t5-small` (generic, non-chemical) to emit a JSON
solvent/catalyst/temperature/yield string from `PRODUCT: ... REACTANTS: ...`. Config:
`--base-model t5-small --learning-rate 5e-4`. Eval with `evaluate_conditions_model_topk.py`
(strict exact/±tolerance + relaxed same-type/same-bucket metrics). Lessons:

- A chemistry-pretrained base (ReactionT5) was tried and abandoned: its 268-token SMILES-only
  vocabulary can't represent JSON/English targets (`<unk>` corruption). `ensure_full_char_coverage`
  adds the missing characters and resizes embeddings **with `mean_resizing=False`** (the default
  multivariate-normal init produced worse-than-random loss on Kaggle's transformers version).
- More clean data helped: 138,869 (300k pool, deduplicated) beat 41,139 on every field.

## Kaggle / Colab workflow

- Run notebooks as **Save & Run All (Commit)**, not an interactive Draft Session (Draft
  sessions reset on tab reload/idle, losing progress).
- Force `machine_shape: NvidiaTeslaT4` in `kernel-metadata.json` — an unpinned GPU can be
  CUDA-incompatible with the installed torch.
- Notebooks clone the repo from GitHub, so a script fix must be pushed before the next run.
- Multi-GPU training uses `torchrun --nproc_per_node=2` for real DDP (plain `python` gives
  slower `DataParallel`). `--time-budget-minutes` force-saves and stops before the session cap.
- Poll with `kaggle kernels status <user>/<slug>`; pull with `kaggle kernels output` once
  `COMPLETE`. `/kaggle/input` is read-only — copy a checkpoint to `/kaggle/working` before an
  eval script that rewrites tokenizer/config in place.

## Development

- Keep changes focused; prefer clear functions over broad abstractions.
- Treat model output as untrusted — validate SMILES with RDKit (`retro_eval.evaluation.is_valid_smiles`).
- Comment only a non-obvious *why* (constraint, workaround, invariant); never restate the code.
- Don't commit secrets, downloaded datasets, `venv/`, or large `data/v2_*` splits (gitignored).
- If adding a dependency, update `pyproject.toml` extras and both READMEs.

## Verification

```bash
python -m pytest tests/          # scoring library; no network/GPU
python -m py_compile scripts/*.py scripts/models/*.py src/retro_eval/*.py
```
