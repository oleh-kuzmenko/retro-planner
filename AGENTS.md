# Agent Notes

## Project Overview

CLI research toolkit comparing 4 retrosynthesis-prediction approaches on the same fixed
100-reaction test set, for a bachelor's thesis. No web app: each approach runs as its own
standalone script, so only one model is ever resident in memory at a time. Every run
writes into a dated, self-describing `experiments/<experiment_id>/<model_slug>/` folder,
and `scripts/aggregate_results.py` discovers every model run under one experiment id (no
hardcoded model list) to build a single comparison CSV.

## Pipeline (see README.MD for full commands)

1. `scripts/build_eval_targets_uspto.py` / `build_eval_targets_ord.py` -- write a fixed
   100-target JSON test set (no Qdrant, no GPU). Every later stage runs against this same
   file.
2. `scripts/models/run_reactiont5.py` -- ReactionT5v2, local HF seq2seq, beam search, no
   reasoning trace. Runs in `colab/02_reactiont5v2.ipynb` (`--device cuda`) or locally on
   CPU.
3. `scripts/models/run_chemllm.py` -- ChemLLM-20B-Chat-SFT GGUF via `llama-cpp-python`,
   zero-shot CoT. Runs in `colab/03_chemllm.ipynb`.
4. `scripts/models/run_qwen_lora_peft.py` -- Qwen2.5-7B-Instruct + this project's own
   trained retrosynthesis LoRA adapter, local HF `peft`. Runs in
   `colab/04_qwen_lora.ipynb` (`--device cuda`) or locally on CPU.
5. `scripts/models/run_rag_cot_llm.py` -- Qdrant hybrid RAG retrieval + Chain-of-Thought
   via a hosted LLM API (GPT-OSS-120B by default), the proposed hybrid system. Runs locally
   (needs a running Qdrant + an API key/base URL for whichever OpenAI-compatible provider
   is passed via `--base-url`/`--api-key`).
6. `scripts/aggregate_results.py` -- combines every model run under one `--experiment-id`
   into `final_aggregated_results.csv`.

`scripts/models/run_chat_zero_shot.py` is a generic OpenAI-compatible chat client (Ollama,
llama.cpp server, ...), usable as a CPU-only alternative to steps 3/4 if a merged/quantized
checkpoint is served locally instead of run in Colab.

RAG (step 5) uses hybrid retrieval from two Qdrant collections:

- `reactions_morgan` - 2048-bit Morgan fingerprints of reaction products.
- `reaction_transforms` - 2048-bit MVP reaction transform fingerprints computed as
  `product_fp XOR combined_reactant_fp`.

Qdrant's own Cosine distance only shortlists ANN candidates; retrieved hits are rescored
with an exact Tanimoto coefficient and merged/reranked in `retrieval.py` with
`weights.molecule * tanimoto_product + weights.reaction * tanimoto_transform +
weights.reaction_class * reaction_class_similarity` (defaults: 0.5 / 0.3 / 0.2) before being
passed to the LLM as CoT context. `reaction_class` biases retrieval toward precedents of the
same disconnection type (see `reaction_classes.py`), not just structurally similar molecules.
`scripts/index_uspto_to_qdrant.py`/`scripts/index_ord_to_qdrant.py` populate both
collections from USPTO-50K/ORD, excluding whatever step 1 already wrote to
`data/*_eval_targets.json` so those targets stay unseen by the RAG index.

## Repository Layout

- `src/retro_eval/` — shared library: chemistry helpers (`chemistry.py`), Qdrant config
  (`config.py`), reaction-class heuristics (`reaction_classes.py`), hybrid retrieval
  (`retrieval.py`), the single-step CoT prompt/repair contract (`prompting.py`,
  `reasoning.py`, `planning.py`), RDKit-based metrics (`evaluation.py`), chat-API providers
  (`providers/chat_api.py`, `providers/retrying.py`), and the eval-scripts' shared
  infrastructure (`harness/`: `records.py` dataset loading, `experiment.py` dated-run-folder
  + crash-safe I/O, `stage_runner.py` the generic per-record inference loop, `parsing.py`
  the CoT/JSON response contracts, `aggregate.py` the run-discovery + CSV-join logic).
- `scripts/sources_uspto.py`, `scripts/sources_ord.py` — per-source parsing/normalization,
  shared by that source's `build_eval_targets_*.py` and `index_*_to_qdrant.py`.
- `scripts/build_eval_targets_uspto.py`, `scripts/build_eval_targets_ord.py` — step 1.
- `scripts/indexing_common.py`, `scripts/index_uspto_to_qdrant.py`,
  `scripts/index_ord_to_qdrant.py` — populate the two Qdrant collections, excluding step
  1's held-out targets.
- `scripts/models/run_reactiont5.py`, `run_chemllm.py`, `run_qwen_lora_peft.py`,
  `run_rag_cot_llm.py`, `run_chat_zero_shot.py` — the model-comparison scripts (steps 2-5).
- `scripts/aggregate_results.py` — step 6.
- `colab/` — GPU notebooks for steps 2-4: clone this repo, install extras, upload the step
  1 JSON, run the matching `scripts/models/run_*.py`, zip+download `experiments/`.
- `tests/` — Pytest suite covering `reasoning`, `prompting`, `retrieval` scoring,
  `evaluation`, and `harness.parsing`/`harness.records` (no network/Qdrant/GPU required).
- `pyproject.toml` — Packaging metadata, dependencies, and `[indexing]`, `[local-models]`,
  `[eval-runner]`, `[test]` extras.
- `docker-compose.yml` — Qdrant service for RAG retrieval; the only container this repo
  needs (no app to containerize).
- `research/fine-tune/v2/` — the notebooks that trained this project's Qwen2.5-7B LoRA
  adapters (data prep, QLoRA training, eval, inference demo). Not required to run anything
  above; kept for methodology citation.
- `experiments/` — gitignored, generated per-run output (see README.MD).
- `README.MD` — setup and usage instructions.
- `venv/` — local virtual environment may exist in the workspace; do not edit or rely on
  committing it.

## Setup

Use Python 3.10+. See README.MD for the exact `pip install -e ".[...]"` commands per extra
(`indexing`, `local-models`, `eval-runner`, `test`). Do not hard-code API keys:
`run_rag_cot_llm.py`/`run_cot_llm.py` talk to any OpenAI-compatible chat-completions
endpoint via `--base-url`/`--api-key`/`--model`. There is no baked-in default provider --
`--base-url` is required -- so pointing at Groq, Cerebras, OpenRouter, Together, Fireworks,
or a local Ollama/llama.cpp server is a flag change, not a code change. The Ollama-backed
`run_chat_zero_shot.py` needs no API key.

Useful environment variables:

- `LLM_API_KEY` — fallback default for `--api-key` on `run_rag_cot_llm.py`/`run_cot_llm.py`.
- `QDRANT_HOST` / `QDRANT_PORT` — default `localhost` / `6333`.

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
  (`USPTO_PAYLOAD_FIELDS` in `sources_uspto.py` / `ORD_PAYLOAD_FIELDS` in `sources_ord.py`,
  via `index_payloads(..., payload_fields=...)`) and written by `build_eval_targets_*.py`
  (via `write_eval_targets(..., fields=...)`): USPTO keeps only
  `product_smiles`/`reactants_smiles` (its conditions are always unknown); ORD additionally
  keeps `reaction_id`, `solvent`, `temperature_celsius`, `catalyst`, and `yield_percent`
  since those are real ORD data.
- USPTO-50K records generally have unknown conditions. ORD records should extract available
  solvents, temperature, catalysts, and yields from protobuf messages, while tolerating
  incomplete or inconsistent records.
- `index_uspto_to_qdrant.py`/`index_ord_to_qdrant.py` exclude by `product_smiles` against
  `--eval-targets-file` (default `data/uspto_eval_targets.json` / `data/ord_eval_targets.json`,
  produced by step 1). Pass `--no-exclude-eval-targets` to index everything regardless. If
  the eval-targets file doesn't exist yet, indexing proceeds with a warning instead of
  failing.
- Treat model output as untrusted. Validate/clean SMILES with RDKit
  (`retro_eval.evaluation.is_valid_smiles`/`is_exact_match_smiles`) before scoring it.
- When changing prompts for the CoT contract, keep the `<think>`/`<reason>` + `<answer>`
  tag contract intact: `retro_eval/reasoning.py` parses `<answer>` as dot-separated reactant
  SMILES, and `retro_eval/planning.py` (`generate_single_step`, used by
  `run_rag_cot_llm.py`) relies on that shape.
- One LLM call produces exactly one retrosynthetic step. Multiple candidate disconnections
  would come from calling the same script's pipeline again (a different temperature or RAG
  context), not from asking the LLM for multiple routes in one response.
- Adding a model to compare: if it's servable over an OpenAI-compatible endpoint (Ollama,
  llama.cpp server, ...), that's a new `run_chat_zero_shot.py` invocation, not a new script.
  Only add a new `scripts/models/run_*.py` file when the model needs bespoke in-process
  loading code (like `run_reactiont5.py`/`run_chemllm.py`/`run_qwen_lora_peft.py` do).
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

For indexing changes, verify against a small limit before a full run:

```bash
docker compose up -d qdrant
python scripts/index_uspto_to_qdrant.py --limit 10
python scripts/index_ord_to_qdrant.py --limit 10
```

The uncapped USPTO-50K + ORD load requires network access to Hugging Face and a running
Qdrant instance. Use `--limit`, ORD allow patterns, or a local ORD data directory for
smaller, repeatable development checks.

## Style

- Use plain Python with type hints where they clarify function contracts.
- Prefer explicit error handling around external services and data parsers: LLM providers
  (Groq, Ollama/OpenAI-compatible), Hugging Face datasets/downloads, Qdrant, ORD protobuf
  parsing, and RDKit parsing.
- Keep files ASCII unless there is a clear reason to preserve existing non-ASCII copy.
- No comments that restate what the code already says; only comment a non-obvious why
  (hidden constraint, workaround, invariant).
