# Agent Notes

## Project Overview

This is a CLI-only Python research toolkit for comparing retrosynthesis-prediction
approaches on a CPU-only, 16GB-RAM laptop, for a bachelor's thesis
(`research/thesis/ПЗ_Кузьменко.pdf`, referenced in code comments as "PZ"). There is no
web app: each approach (currently 4, extensible) runs as its own standalone script under
`scripts/models/`, so only one model is ever resident in memory at a time. Every run writes
into a dated, self-describing `experiments/<experiment_id>/<model_slug>/` folder, and
`scripts/aggregate_results.py` discovers every model run under one experiment id (no
hardcoded model list) to build a single comparison CSV.

The 4 current approaches:

1. **ReactionT5v2** (`run_reactiont5.py`) — local HF seq2seq, beam search, no reasoning trace.
2. **A quantized chat LLM** (e.g. ChemLLM GGUF) served via Ollama, zero-shot CoT
   (`run_chat_zero_shot.py`).
3. **Qwen2.5-7B-Instruct + a retrosynthesis LoRA adapter** — local HF `peft` on CPU
   (`run_qwen_lora_peft.py`), or the same adapter merged/quantized and served via Ollama
   (`run_chat_zero_shot.py --response-format json`).
4. **Qdrant hybrid RAG retrieval + Chain-of-Thought via Groq's Llama-3.3-70B**
   (`run_rag_cot_groq.py`) — the proposed hybrid system.

RAG uses hybrid retrieval from two Qdrant collections:

- `reactions_morgan` - 2048-bit Morgan fingerprints of reaction products.
- `reaction_transforms` - 2048-bit MVP reaction transform fingerprints computed as
  `product_fp XOR combined_reactant_fp`.

Qdrant's own Cosine distance only shortlists ANN candidates; retrieved hits are rescored
with an exact Tanimoto coefficient and merged/reranked in `retrieval.py` with
`weights.molecule * tanimoto_product + weights.reaction * tanimoto_transform` by default
(`reaction_class` similarity is an opt-in extension, weight 0.0 unless
`EXPERIMENTAL_RETRIEVAL_WEIGHTS` is used) before being passed to the LLM as CoT context.
`scripts/index_uspto_to_qdrant.py`/`scripts/index_ord_to_qdrant.py` populate both
collections from USPTO-50K and Open Reaction Database (ORD) reactions, holding out ~100
target molecules per source for evaluation.

## Repository Layout

- `src/retro_eval/` — shared library: chemistry helpers (`chemistry.py`), Qdrant config
  (`config.py`), reaction-class heuristics (`reaction_classes.py`), hybrid retrieval
  (`retrieval.py`), the single-step CoT prompt/repair contract (`prompting.py`,
  `reasoning.py`, `planning.py`), RDKit-based metrics (`evaluation.py`), chat-API providers
  (`providers/chat_api.py`, `providers/retrying.py`), and the eval-scripts' shared
  infrastructure (`harness/`: `records.py` dataset loading, `experiment.py` dated-run-folder
  + crash-safe I/O, `stage_runner.py` the generic per-record inference loop, `parsing.py`
  the CoT/JSON response contracts, `aggregate.py` the run-discovery + CSV-join logic).
- `scripts/indexing_common.py`, `scripts/index_uspto_to_qdrant.py`,
  `scripts/index_ord_to_qdrant.py` — populate the two Qdrant collections from USPTO-50K/ORD.
- `scripts/models/run_reactiont5.py`, `run_chat_zero_shot.py`, `run_qwen_lora_peft.py`,
  `run_rag_cot_groq.py` — the 4 model-comparison scripts.
- `scripts/aggregate_results.py` — combines every model run under one `--experiment-id`
  into `final_aggregated_results.csv`.
- `tests/` — Pytest suite covering `reasoning`, `prompting`, `retrieval` scoring,
  `evaluation`, and `harness.parsing`/`harness.records` (no network/Qdrant/GPU required).
- `pyproject.toml` — Packaging metadata, dependencies, and `[indexing]`, `[local-models]`,
  `[eval-runner]`, `[test]` extras.
- `docker-compose.yml` — Qdrant service for RAG retrieval; the only container this repo
  needs (no app to containerize).
- `research/` — prior research artifacts kept for methodology citation: the thesis
  (`research/thesis/`), a manual/qualitative eval journal (`research/eval.md`), and the
  fine-tuning notebooks/data for the Qwen LoRA adapters (`research/fine-tune/`,
  `research/base_models_res/`). Not required to run anything above.
- `experiments/` — gitignored, generated per-run output (see README.MD).
- `README.MD` — setup and usage instructions.
- `venv/` — local virtual environment may exist in the workspace; do not edit or rely on
  committing it.

## Setup

Use Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

The indexing scripts have extra dependencies (Hugging Face dataset/download helpers,
`ord-schema` for ORD protobuf parsing):

```bash
pip install -e ".[indexing]"
```

Every `scripts/models/run_*.py` script needs `tenacity`/`tqdm`:

```bash
pip install -e ".[eval-runner]"
```

`run_reactiont5.py` and `run_qwen_lora_peft.py` additionally need heavy ML dependencies:

```bash
pip install -e ".[local-models]"
```

Test dependencies (`pytest`):

```bash
pip install -e ".[test]"
```

Do not hard-code API keys. `run_rag_cot_groq.py` accepts `--groq-api-key` and falls back to
`GROQ_API_KEY`; the Ollama-backed scripts (`run_chat_zero_shot.py`, and
`run_qwen_lora_peft.py`'s merged/GGUF alternative) need no API key.

Useful environment variables:

- `GROQ_API_KEY` — used by `run_rag_cot_groq.py` if `--groq-api-key` isn't passed.
- `QDRANT_HOST` - Qdrant host, currently `localhost` if unset.
- `QDRANT_PORT` - Qdrant port, currently `6333` if unset.

## Common Commands

Start Qdrant:

```bash
docker compose up -d qdrant
```

Index a small sample from USPTO-50K first:

```bash
python scripts/index_uspto_to_qdrant.py --limit 100 --recreate
```

Or from ORD:

```bash
python scripts/index_ord_to_qdrant.py --limit 100 --recreate
```

Run one model-comparison stage (see README.MD for all 4 + aggregation):

```bash
python scripts/models/run_reactiont5.py --input data/uspto_eval_targets.json --limit 10
python scripts/aggregate_results.py --input data/uspto_eval_targets.json
```

Each indexing script always drops and recreates both collections for its own source;
`--recreate` is accepted for backwards compatibility. Since both scripts default to the
same `reactions_morgan`/`reaction_transforms` collection names, running one after the other
replaces rather than merges data — point `--collection`/`--transform-collection` at
different names if you need both sources queryable at once. Use `--limit 0` for all
remaining reactions, `--ord-data-dir /path/to/ord-data` and
`--ord-allow-pattern "data/4d/*.pb.gz"` for narrower ORD dev runs, and
`--eval-targets-count`/`--eval-targets-file` to control how many target molecules are set
aside (default 100) instead of indexed, so they stay unseen for evaluation.

## Development Guidance

- Keep changes focused. This project is currently a compact prototype, so prefer clear
  functions over broad abstractions.
- There is no UI to preserve — every user-facing surface is a CLI script's `argparse`
  interface plus its logged output and `experiments/` files.
- RAG mode searches `reactions_morgan` and `reaction_transforms` with 2048-bit Morgan-based
  vectors. Keep vector size and fingerprint generation aligned between
  `src/retro_eval/chemistry.py` and `scripts/indexing_common.py`.
- Each indexer normalizes its source into a full in-memory record, but only a
  source-specific field subset is actually stored on the Qdrant point
  (`USPTO_PAYLOAD_FIELDS` / `ORD_PAYLOAD_FIELDS`, via `index_payloads(...,
  payload_fields=...)`) and written to `--eval-targets-file` (via `write_eval_targets(...,
  fields=...)`): USPTO keeps only `product_smiles`/`reactants_smiles` (its conditions are
  always unknown); ORD additionally keeps `reaction_id`, `solvent`, `temperature_celsius`,
  `catalyst`, and `yield_percent` since those are real ORD data.
- USPTO-50K records generally have unknown conditions. ORD records should extract available
  solvents, temperature, catalysts, and yields from protobuf messages, while tolerating
  incomplete or inconsistent records.
- Both indexers set aside the first `--eval-targets-count` (default 100) target molecules
  into `--eval-targets-file` instead of indexing them, so those molecules stay unseen and
  can be used as evaluation targets without leaking into the RAG index.
- Treat model output as untrusted. Validate/clean SMILES with RDKit
  (`retro_eval.evaluation.is_valid_smiles`/`is_exact_match_smiles`) before scoring it.
- When changing prompts for the CoT contract, keep the `<think>`/`<reason>` + `<answer>`
  tag contract intact: `retro_eval/reasoning.py` parses `<answer>` as dot-separated reactant
  SMILES, and `retro_eval/planning.py` (`generate_single_step`, used by
  `run_rag_cot_groq.py`) relies on that shape.
- One LLM call produces exactly one retrosynthetic step. Multiple candidate disconnections
  would come from calling the same script's pipeline again (a different temperature or RAG
  context), not from asking the LLM for multiple routes in one response.
- Adding a 5th/6th model to compare: if it's servable over an OpenAI-compatible endpoint
  (Ollama, llama.cpp server, ...), that's a new `run_chat_zero_shot.py --model-slug ...`
  invocation, not a new script. Only add a new `scripts/models/run_*.py` file when the model
  needs bespoke in-process loading code (like `run_reactiont5.py`/`run_qwen_lora_peft.py`
  do).
- When changing molecule handling, prefer RDKit APIs over manual SMILES parsing. Existing
  abbreviation replacement in `clean_and_canonicalize` is intentionally small and heuristic.
- Avoid adding secrets, downloaded datasets, Qdrant storage, virtual environments, or
  `experiments/` run output to git.
- If adding dependencies, update both `pyproject.toml` extras and the setup docs
  (README.MD/this file).

## Verification

For code changes, run syntax checks and the test suite:

```bash
python -m py_compile scripts/*.py scripts/models/*.py src/retro_eval/*.py src/retro_eval/harness/*.py src/retro_eval/providers/*.py
python -m pytest tests/
```

The test suite has no network/Qdrant/GPU dependency, so it must pass without any external
services running.

For a change to one of the `scripts/models/run_*.py` scripts, do a small end-to-end smoke
test before considering it done: run it with `--limit 2-3` against a tiny input JSON and
confirm `experiments/<experiment_id>/<model_slug>/{results.json,inference_logs.json,
run_meta.json}` are written and well-formed, then run `scripts/aggregate_results.py` against
that experiment id and confirm the CSV is produced (including when only some model runs
exist — it should warn about missing ones, not crash).

For indexing changes, verify against a small limit before a full run (pass
`--eval-targets-count 0` so a tiny `--limit` isn't entirely consumed by the hold-out):

```bash
docker compose up -d qdrant
python scripts/index_uspto_to_qdrant.py --limit 10 --eval-targets-count 0 --recreate
python scripts/index_ord_to_qdrant.py --limit 10 --eval-targets-count 0 --recreate
```

The uncapped USPTO-50K + ORD load requires network access to Hugging Face and a running
Qdrant instance. Use `--limit`, ORD allow patterns, or a local ORD data directory for
smaller, repeatable development checks.

## Style

- Use plain Python with type hints where they clarify function contracts.
- Prefer explicit error handling around external services and data parsers: LLM providers
  (Groq, Ollama/OpenAI-compatible), Hugging Face datasets/downloads, Qdrant, ORD protobuf
  parsing, and RDKit parsing.
- Keep files ASCII unless there is a clear reason to preserve existing non-ASCII copy (the
  `research/` materials are in Ukrainian; that's expected).
