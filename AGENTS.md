# Agent Notes

## Project Overview

This is a small Python/Streamlit retrosynthesis planner. The main app lets a user draw a target molecule with Ketcher or enter SMILES directly, canonicalizes the molecule with RDKit, optionally retrieves similar reactions from Qdrant, builds a 4-block `[System]/[Context]/[Instruction]/[Input]` chain-of-thought prompt, calls one registered `LLMProvider`, parses the plain-text `<think>`/`<reason>` + `<answer>` response (no JSON), chemically validates the returned precursors, and renders the single retrosynthetic step (with an option to generate another candidate step for the same target).

RAG currently uses hybrid retrieval from two Qdrant collections:

- `reactions_morgan` - 2048-bit Morgan fingerprints of reaction products.
- `reaction_transforms` - 2048-bit MVP reaction transform fingerprints computed as `product_fp XOR combined_reactant_fp`.

Qdrant's own Cosine distance only shortlists ANN candidates; retrieved hits are rescored with an exact Tanimoto coefficient and merged/reranked in `retrieval.py` with `weights.molecule * tanimoto_product + weights.reaction * tanimoto_transform` by default (`reaction_class` similarity is an opt-in extension, weight 0.0 unless `EXPERIMENTAL_RETRIEVAL_WEIGHTS` is used) before being passed to the LLM provider as `[Context]` examples. There is also a Qdrant indexing utility under `scripts/` for rebuilding both collections from USPTO-50K and Open Reaction Database (ORD) reactions.

## Repository Layout

- `app.py` - Thin Streamlit entrypoint that calls `retro_planner.app.main`.
- `src/retro_planner/` - Application package: Streamlit UI (`app.py`, `streamlit_views.py`), chemistry helpers (`chemistry.py`, `rendering.py`), retrieval (`retrieval.py`, `reaction_classes.py`, `config.py`), single-step orchestration (`planning.py`), prompt construction (`prompting.py`), response parsing/validation (`reasoning.py`), evaluation metrics (`evaluation.py`), and the `providers/` package.
- `src/retro_planner/providers/` - `LLMProvider` protocol and registry (`__init__.py`), chat-API backends (`chat_api.py`: Groq/OpenAI/custom OpenAI-compatible), and local/research backends (`local_seq2seq.py`, `local_causal.py`, `local_gguf.py`) behind the `[local-models]` extras.
- `scripts/index_uspto50k_to_qdrant.py` - CLI script for recreating Qdrant collections and indexing USPTO-50K plus ORD reactions.
- `scripts/evaluate_retrosynthesis.py` - CLI for automated Top-k / Structure Success Rate evaluation on USPTO-50K (zero-shot vs RAG+CoT, any registered provider).
- `tests/` - Pytest suite covering `reasoning`, `prompting`, `retrieval` scoring, `providers`, and `evaluation` (no network/Qdrant/GPU required).
- `pyproject.toml` - Packaging metadata, dependencies, and `[indexing]`, `[local-models]`, `[test]` extras.
- `Dockerfile` - Container for running the Streamlit app.
- `docker-compose.yml` - Qdrant service for local vector search work.
- `README.MD` - User setup and app usage instructions.
- `venv/` - Local virtual environment may exist in the workspace; do not edit or rely on committing it.

## Setup

Use Python 3.10+ for local app work.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

The indexing script has extra dependencies, including Hugging Face dataset/download helpers and `ord-schema` for ORD protobuf parsing:

```bash
pip install -e ".[indexing]"
```

Local/research model providers (`local_seq2seq`, `local_causal`, `local_gguf`) need heavy ML dependencies, kept out of the base install:

```bash
pip install -e ".[local-models]"
```

Test dependencies (`pytest`):

```bash
pip install -e ".[test]"
```

Do not hard-code API keys. Each cloud provider in `LLM_PROVIDER_REGISTRY` (`providers/chat_api.py`) accepts its key/base URL from the Streamlit sidebar and falls back to that provider's environment variable (`GROQ_API_KEY`, `OPENAI_API_KEY`, `CUSTOM_LLM_API_KEY`, ...) as the default field value. Local providers do not require an API key.

Useful environment variables:

- `GROQ_MODEL` / `OPENAI_MODEL` / `CUSTOM_LLM_MODEL` - default model shown in the sidebar for each chat-API provider.
- `CUSTOM_LLM_BASE_URL` - base URL for the custom OpenAI-compatible provider (e.g. a local Ollama endpoint).
- `QDRANT_HOST` - Qdrant host, currently `localhost` if unset.
- `QDRANT_PORT` - Qdrant port, currently `6333` if unset.

## Common Commands

Run the app locally:

```bash
streamlit run app.py
```

Build and run the app with Docker:

```bash
docker build -t retro-planner .
docker run -p 8501:8501 retro-planner
```

Start Qdrant for indexing/search experiments:

```bash
docker compose up -d qdrant
```

Index a small sample from the default USPTO-50K + ORD sources first:

```bash
python scripts/index_uspto50k_to_qdrant.py --limit 100 --recreate
```

Index up to 500,000 reactions from the configured sources into the default `reactions_morgan` and `reaction_transforms` collections:

```bash
python scripts/index_uspto50k_to_qdrant.py --recreate
```

The indexer always drops and recreates both collections; `--recreate` is accepted for backwards compatibility. Use `--limit 0` for all available reactions, or `--sources uspto`, `--sources ord`, `--ord-data-dir /path/to/ord-data`, and `--ord-allow-pattern "data/4d/*.pb.gz"` for narrower development runs.

## Development Guidance

- Keep changes focused. This project is currently a compact prototype, so prefer clear functions over broad abstractions.
- Preserve the user-facing Streamlit workflow unless the requested task explicitly changes it: draw or enter molecule, validate SMILES, optionally retrieve RAG context, call the LLM provider, render the single-step reasoning and reaction image.
- RAG mode searches `reactions_morgan` and `reaction_transforms` with 2048-bit Morgan-based vectors. Keep vector size and fingerprint generation aligned between `src/retro_planner/chemistry.py` and `scripts/index_uspto50k_to_qdrant.py`.
- The indexer writes a shared payload schema with `product_smiles`, `reactant_smiles`, `reactants_smiles`, `conditions`, and `source`. Preserve legacy fields such as `solvent`, `temperature_celsius`, `yield_percent`, and `reactants_smiles` because `retrieval.py` and `prompting.py` still read them.
- USPTO-50K records generally have unknown conditions. ORD records should extract available solvents, temperature, catalysts, and yields from protobuf messages, while tolerating incomplete or inconsistent records.
- Treat model output as untrusted. Validate/clean SMILES with RDKit before rendering or using returned reactants.
- When changing prompts, keep the `<think>`/`<reason>` + `<answer>` tag contract intact: `retro_planner/reasoning.py` parses `<answer>` as dot-separated reactant SMILES, and `retro_planner/planning.py` (`generate_single_step`) relies on that shape. This is plain tagged text, not JSON.
- One LLM call produces exactly one retrosynthetic step (`StepResult`). Multiple candidate disconnections are obtained by calling `generate_single_step` again (e.g. a different temperature or RAG context), not by asking the LLM for multiple routes in one response.
- When changing molecule handling, prefer RDKit APIs over manual SMILES parsing. Existing abbreviation replacement in `clean_and_canonicalize` is intentionally small and heuristic.
- Avoid adding secrets, downloaded datasets, Qdrant storage, or virtual environments to git.
- If adding dependencies, update both local setup documentation and Docker installation instructions.

## Verification

For code changes, run syntax checks and the test suite:

```bash
python -m py_compile app.py scripts/index_uspto50k_to_qdrant.py src/retro_planner/*.py src/retro_planner/providers/*.py
python -m pytest tests/
```

The test suite has no network/Qdrant/GPU dependency, so it must pass without any external services running.

For UI changes, run Streamlit and manually verify:

- default aspirin molecule validates;
- missing provider API key (or base URL for the custom provider) shows a warning instead of calling the API;
- invalid structures show an error instead of calling the API;
- successful model responses render the "Chemist's Reasoning" expander (the `<think>` text), the precursors/product codes, and either a reaction image or a clear warning;
- "Generate another candidate" appends a new `StepResult` without re-querying Qdrant;
- RAG mode can fall back gracefully if Qdrant or `reaction_transforms` is unavailable.

For indexing changes, verify against a small limit before a full run:

```bash
docker compose up -d qdrant
python scripts/index_uspto50k_to_qdrant.py --limit 10 --recreate
```

The uncapped USPTO-50K + ORD load requires network access to Hugging Face and a running Qdrant instance. Use `--limit`, ORD allow patterns, or a local ORD data directory for smaller, repeatable development checks.

## Style

- Use plain Python with type hints where they clarify function contracts.
- Keep Streamlit labels and messages concise and practical for chemists.
- Prefer explicit error handling around external services and data parsers: LLM providers (Groq, OpenAI, custom endpoints, local models), Hugging Face datasets/downloads, Qdrant, ORD protobuf parsing, and RDKit parsing.
- Keep files ASCII unless there is a clear reason to preserve existing non-ASCII UI copy.
