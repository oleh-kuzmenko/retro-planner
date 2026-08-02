# retro-planner

*[Українською](README.uk.md)*

Intelligent system for **retrosynthesis of organic compounds**. Two independently
trained models, both on the Open Reaction Database (ORD), with USPTO-50K kept as an
independent generalization test:

- **Model 1 — reactant prediction.** A fine-tuned `ReactionT5v2` (chemistry-pretrained
  T5) that predicts reactant SMILES from a product SMILES.
- **Model 2 — reaction-condition prediction.** A `t5-small` that predicts solvent,
  catalyst, temperature and yield for a product/reactants pair.

Full methodology and every measured number are in **[RESULTS.md](RESULTS.md)**. Headline:
Model 1 reaches **82.3% top-5 `core_exact_match` on ORD** by merging the beam candidates of
two independently fine-tuned checkpoints; Model 2 improves on every field when trained on
138,869 (vs 41,139) deduplicated, leak-free reactions.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Data building, training and evaluation entry points |
| `scripts/models/` | Model 1 inference/eval: top-k beam search, ensemble merge, round-trip rerank |
| `src/retro_eval/` | Library: SMILES canonicalization, exact/core-match scoring, condition similarity |
| `kaggle/`, `colab/` | GPU notebooks that run the training/eval scripts |
| `experiments/` | Committed evaluation results (JSON) — see `experiments/README.md` |
| `tests/` | Unit tests for the scoring library |
| `RESULTS.md` | The report: all tables and conclusions |

## Pipeline

Large derived splits under `data/` are gitignored and regenerated deterministically
(fixed seed); only the small eval-target JSONs are committed.

**1. Build data** (needs ORD via Hugging Face; no GPU):

```bash
pip install -e ".[indexing]"
python scripts/build_eval_targets_ord.py                  # Model 1 held-out test (data/v2_ord_eval_targets.json)
python scripts/build_train_data_ord.py --pool-count 60000 # reactant + condition train/val(/test) splits
python scripts/build_root_aligned_data.py \               # optional: root-aligned reactant target (Model 1 variant 6)
    --input data/v2_ord_train/reactants_train.jsonl \
    --output data/v2_ord_train_rootaligned/reactants_train.jsonl
```

**2. Train** (GPU — via the `kaggle/`/`colab/` notebooks, which call these scripts):

```bash
# Model 1 (reactant): fine-tune ReactionT5v2 on ORD
python scripts/train_reactant_model_ord.py --no-augment --learning-rate 5e-5 --num-train-epochs 3 ...
# Model 2 (conditions): train t5-small on ORD
python scripts/train_conditions_model.py --base-model t5-small --learning-rate 5e-4 ...
```

**3. Evaluate**:

```bash
# Model 1: top-1/3/5 exact & core-exact
python scripts/models/run_reactiont5_topk.py --input data/v2_ord_eval_targets.json --t5-model <ckpt> --num-beams 10 --output out.json
# merge two checkpoints' candidates (the ensemble)
python scripts/models/ensemble_topk.py --primary a.json --secondary b.json --output ensemble.json
# Model 2: per-field strict & relaxed metrics
python scripts/evaluate_conditions_model_topk.py --model-dir <ckpt> --test-file data/v2_ord_train_300k/conditions_test.jsonl --num-beams 10 --output out.json
```

## Local setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[local-models,indexing]"   # torch/transformers + ORD data tooling
pip install -e ".[test]"                     # pytest
```

`build_root_aligned_data.py` additionally needs `pip install rxnmapper "setuptools<81"`.
Linux: `sudo apt-get install libxrender1 libxext6 libgl1` if RDKit can't find GUI libs.

## Tests

```bash
python -m pytest tests/
```

No network/GPU dependency.
