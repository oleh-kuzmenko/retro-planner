# retro-eval

A CLI research toolkit for comparing retrosynthesis-prediction approaches on a CPU-only,
16GB-RAM laptop: local Hugging Face models (ReactionT5v2, Qwen2.5-7B + LoRA), a quantized
chat LLM served locally via Ollama, and a Qdrant-RAG + Chain-of-Thought pipeline called
through the Groq API. Each approach runs as its own standalone script, so only one model
is ever resident in memory at a time. Results, logs, and prompts from every run are written
into a dated `experiments/<experiment_id>/<model_slug>/` folder for easy comparison and
citation.

## Prerequisites

* **Python 3.10+**.
* **Qdrant** for RAG retrieval. The included `docker-compose.yml` starts local Qdrant on
  ports `6333`/`6334`.
* **A Groq API key** (`GROQ_API_KEY`) for the RAG+CoT script — free tier is enough for
  small evaluation runs.
* **Ollama** (or an equivalent OpenAI-compatible server) running locally if you want to
  test a quantized chat model (e.g. a GGUF checkpoint) zero-shot.
* 16GB RAM is enough to run each model **one at a time**; running several simultaneously
  will not fit.

## 1. Set Up the Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Install the extras you need:

```bash
pip install -e ".[eval-runner]"    # tenacity + tqdm -- needed by every scripts/models/run_*.py script
pip install -e ".[local-models]"   # torch, transformers, peft -- needed only by run_reactiont5.py and run_qwen_lora_peft.py
pip install -e ".[indexing]"       # datasets, huggingface_hub, ord-schema -- needed only to populate Qdrant
pip install -e ".[test]"           # pytest
```

Linux users: if RDKit needs GUI libraries it doesn't find, install:

```bash
sudo apt-get install libxrender1 libxext6 libgl1
```

## 2. Start Qdrant

```bash
docker compose up -d qdrant
```

## 3. Index Reaction Examples for RAG

There are two separate indexer scripts, one per source: `scripts/index_uspto_to_qdrant.py`
(USPTO-50K, via `datasets`) and `scripts/index_ord_to_qdrant.py` (Open Reaction Database,
via `ord-schema`). Each one rebuilds its own vector collections from scratch — it drops and
recreates both `--collection`/`--transform-collection` before loading data — and, before
indexing, sets aside the first `--eval-targets-count` (default 100) target molecules into
`--eval-targets-file` (JSON) instead of indexing them, so those molecules stay unseen and
can be used as held-out evaluation targets for the model-comparison scripts below, without
leaking into the RAG index. Because both scripts default to the same
`reactions_morgan`/`reaction_transforms` collection names, running one after the other
**replaces** rather than merges data; pass different `--collection`/`--transform-collection`
names if you need both sources queryable at once.

Index a limited sample from USPTO-50K:

```bash
python scripts/index_uspto_to_qdrant.py --limit 100 --recreate
```

Index a limited sample from ORD:

```bash
python scripts/index_ord_to_qdrant.py --limit 100 --recreate
```

By default, each script indexes up to 500,000 reactions (after the eval-target hold-out):

```bash
python scripts/index_uspto_to_qdrant.py --recreate
python scripts/index_ord_to_qdrant.py --recreate
```

To process all remaining reactions instead of using the default cap:

```bash
python scripts/index_uspto_to_qdrant.py --limit 0 --recreate
python scripts/index_ord_to_qdrant.py --limit 0 --recreate
```

`index_uspto_to_qdrant.py` downloads `pingzhili/uspto-50k` through `datasets`;
`index_ord_to_qdrant.py` downloads ORD protobuf files from the Hugging Face mirror
`Open-Reaction-Database/ord-data` unless `--ord-data-dir` is provided.

To use a local ORD checkout or a single ORD protobuf file:

```bash
python scripts/index_ord_to_qdrant.py --ord-data-dir /path/to/ord-data
python scripts/index_ord_to_qdrant.py --ord-data-dir /path/to/sample.pb.gz
```

To download only a subset of ORD from Hugging Face:

```bash
python scripts/index_ord_to_qdrant.py --ord-allow-pattern "data/4d/*.pb.gz"
```

To change how many target molecules are set aside for evaluation, or where they're written:

```bash
python scripts/index_uspto_to_qdrant.py --eval-targets-count 100 --eval-targets-file data/uspto_eval_targets.json
python scripts/index_ord_to_qdrant.py --eval-targets-count 100 --eval-targets-file data/ord_eval_targets.json
```

`--eval-targets-file` is a plain JSON array — exactly the input format the model-comparison
scripts below expect. Each record has `product_smiles` (the target to predict reactants
for) and `reactants_smiles` (the dot-separated reference precursor set to compare a
prediction against), plus whatever other fields that source actually has data for.

`data/uspto_eval_targets.json` — USPTO-50K has no real conditions/class data past
`product_smiles`/`reactants_smiles`, so that's all that's kept:

```json
[
  {
    "product_smiles": "CC(C)(C)OC(=O)N1CCC(NC(=O)c2ccccc2)CC1",
    "reactants_smiles": "CC(C)(C)OC(=O)N1CCC(N)CC1.O=C(Cl)c1ccccc1"
  }
]
```

`data/ord_eval_targets.json` — ORD protobuf records genuinely carry a reaction id and
conditions, so those are kept too:

```json
[
  {
    "reaction_id": "ord-e986ebcd9edb45428124e738770f3eaa",
    "product_smiles": "O=C(/C=C/c1ccccc1)c1ccccc1",
    "reactants_smiles": "CC(=O)c1ccccc1.O=Cc1ccccc1",
    "solvent": "CCO, O",
    "temperature_celsius": 25.0,
    "catalyst": null,
    "yield_percent": 92.0
  }
]
```

Each creates the same two Qdrant collections:

* `reactions_morgan`: product Morgan fingerprints for molecule similarity.
* `reaction_transforms`: MVP reaction fingerprints computed as `product_fp XOR reactant_fp`.

## 4. Run and Compare Model Approaches

Each approach is its own standalone script under `scripts/models/`, so you can run them one
at a time, on different days if memory or time is tight, and add a new approach later
without touching the existing scripts.

**Stage 1 — ReactionT5v2** (local HF seq2seq, beam search, no reasoning trace):

```bash
pip install -e ".[local-models,eval-runner]"
python scripts/models/run_reactiont5.py --input data/uspto_eval_targets.json --limit 25
```

**Stage 2 — a quantized chat LLM (e.g. ChemLLM GGUF) via Ollama, zero-shot:**

```bash
ollama serve
# ollama create/pull whatever GGUF you're testing, e.g. under the tag "chemllm"
python scripts/models/run_chat_zero_shot.py --input data/uspto_eval_targets.json \
    --model-slug chemllm_ollama --model chemllm --response-format cot --limit 25
```

**Stage 3 — Qwen2.5-7B-Instruct + a retrosynthesis LoRA adapter**, local HF `peft` on CPU by
default:

```bash
python scripts/models/run_qwen_lora_peft.py --input data/uspto_eval_targets.json --limit 25
```

If Qwen2.5-7B doesn't fit in memory or is too slow on CPU, merge the LoRA into the base
model, quantize to GGUF, serve it via Ollama, and run the same generic chat script instead
(same JSON reactants-prediction contract):

```bash
python scripts/models/run_chat_zero_shot.py --input data/uspto_eval_targets.json \
    --model-slug qwen25_7b_lora_ollama --model <your-ollama-tag> --response-format json
```

**Stage 4 — Qdrant hybrid RAG retrieval + Chain-of-Thought via Groq's Llama-3.3-70B**
(the proposed hybrid system):

```bash
GROQ_API_KEY=... python scripts/models/run_rag_cot_groq.py --input data/uspto_eval_targets.json --limit 25
```

Passing an ORD-format input file (`data/ord_eval_targets.json`) also asks the model to
propose reaction conditions (solvent/temperature/catalyst) — logged for qualitative review
only, never scored programmatically.

**Aggregate every model run into one comparison table:**

```bash
python scripts/aggregate_results.py --input data/uspto_eval_targets.json
```

### Where results go

Every `scripts/models/run_*.py` script writes into
`experiments/<experiment_id>/<model_slug>/`:

* `results.json` — one row per record: product, ground-truth reactants, predicted
  reactants.
* `inference_logs.json` — the exact request sent and raw response received for every
  record, for qualitative citation.
* `run_meta.json` — the CLI arguments used (API keys redacted), the input file's sha256,
  record count, and start/finish timestamps/status, so a run is fully self-describing.

`experiment_id` defaults to today's date (`YYYY-MM-DD`), overridable with `--experiment-id`
so stages run on different days (a likely scenario given the memory constraints) still group
into one comparison. `scripts/aggregate_results.py` discovers every `<model_slug>/` folder
under an experiment id automatically — nothing is hardcoded to "4 stages" — and writes
`experiments/<experiment_id>/final_aggregated_results.csv`: one row per target with the
ground truth plus, for every discovered model, its predicted reactants, RDKit validity, and
exact match (canonical SMILES set comparison) against the reference.

`/experiments/` is gitignored, same as `/data/` — these are generated run artifacts, not
source.

## Development Checks

```bash
python -m py_compile scripts/*.py scripts/models/*.py src/retro_eval/*.py src/retro_eval/harness/*.py src/retro_eval/providers/*.py
pip install -e ".[test]"
python -m pytest tests/
```

The test suite (`tests/`) covers response parsing/validation, prompt construction,
retrieval scoring, and the evaluation metrics without needing Qdrant, network access, or
GPU/local-model dependencies.

For indexing changes, run a limited indexing check:

```bash
docker compose up -d qdrant
python scripts/index_uspto_to_qdrant.py --limit 10 --eval-targets-count 0 --recreate
python scripts/index_ord_to_qdrant.py --limit 10 --eval-targets-count 0 --recreate
```

The uncapped USPTO-50K + ORD load requires network access to Hugging Face and a running
Qdrant instance. Use `--limit`, `--ord-allow-pattern`, or `--ord-data-dir` for smaller,
repeatable development runs.

## Research Materials

`research/` holds the prior research this toolkit builds on: `research/thesis/` (the
bachelor's thesis this project implements, plus its architecture diagrams),
`research/eval.md` (a manual/qualitative evaluation journal scoring 6 earlier
prompting/model configurations across ~10 molecules), `research/fine-tune/` (the notebooks
and data used to fine-tune the Qwen2.5-7B LoRA adapters), and `research/base_models_res/`
(exploratory notebook for the ChemLLM GGUF model). None of it is required to run the
scripts above; it's kept for methodology citation and reproducibility.
