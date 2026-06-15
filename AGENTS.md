# Agent Notes

## Project Overview

This is a small Python/Streamlit retrosynthesis planner. The main app lets a user draw a target molecule with Ketcher or enter SMILES directly, canonicalizes the molecule with RDKit, optionally retrieves similar reactions from Qdrant, sends the target SMILES plus retrieved context to Groq's Llama model, expects multi-route retrosynthesis JSON, and renders route steps/reaction images with RDKit.

RAG currently uses hybrid retrieval from two Qdrant collections:

- `reactions_morgan` - 2048-bit Morgan fingerprints of reaction products.
- `reaction_transforms` - 2048-bit MVP reaction transform fingerprints computed as `product_fp XOR combined_reactant_fp`.

Retrieved hits are merged and reranked with product similarity, transform similarity, and inferred reaction-class similarity before being passed to Groq as context. There is also a Qdrant indexing utility under `scripts/` for loading USPTO-50K reactions into both collections.

## Repository Layout

- `app.py` - Thin Streamlit entrypoint that calls `retro_planner.app.main`.
- `src/retro_planner/` - Application package with Streamlit UI, chemistry helpers, LLM providers, prompts, planning, rendering, and retrieval logic.
- `scripts/index_uspto50k_to_qdrant.py` - CLI script for indexing USPTO-50K reactions into Qdrant.
- `tests/` - Unit tests for chemistry, prompts, planning, retrieval, reaction class inference, and indexer row normalization.
- `pyproject.toml` - Packaging metadata, dependencies, optional extras, and pytest configuration.
- `Dockerfile` - Container for running the Streamlit app.
- `docker-compose.yml` - Qdrant service for local vector search work.
- `README.MD` - User setup and app usage instructions.
- `testing/test-reactions` - Manual golden-product evaluation notes for no-RAG, product-RAG, and hybrid-RAG comparisons.
- `venv/` - Local virtual environment may exist in the workspace; do not edit or rely on committing it.

## Setup

Use Python 3.10+ for local app work.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

The indexing script has extra dependencies:

```bash
pip install -e ".[indexing]"
```

Install test dependencies for development:

```bash
pip install -e ".[dev]"
```

Do not hard-code API keys. The app asks for the Groq API key in the Streamlit sidebar and also accepts `GROQ_API_KEY` from the environment as the default field value.

Useful environment variables:

- `GROQ_MODEL` - default model shown in the sidebar, currently `llama-3.3-70b-versatile` if unset.
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

Index a small sample first:

```bash
python scripts/index_uspto50k_to_qdrant.py --limit 100 --recreate
```

Index the configured dataset into the default `reactions_morgan` and `reaction_transforms` collections:

```bash
python scripts/index_uspto50k_to_qdrant.py --recreate
```

## Development Guidance

- Keep changes focused. This project is currently a compact prototype, so prefer clear functions over broad abstractions.
- Preserve the user-facing Streamlit workflow unless the requested task explicitly changes it: draw or enter molecule, validate SMILES, optionally retrieve RAG context, call Groq, render routes and step images.
- RAG mode searches `reactions_morgan` and `reaction_transforms` with 2048-bit Morgan-based vectors. Keep vector size and fingerprint generation aligned between `src/retro_planner/chemistry.py` and `scripts/index_uspto50k_to_qdrant.py`.
- Treat model output as untrusted. Validate/clean SMILES with RDKit before rendering or using returned reactants.
- When changing Groq prompts, keep the JSON-only contract intact because `get_retrosynthesis_plan` and `call_groq_with_rag` parse responses with `json.loads`.
- Preserve the current multi-route schema where possible: `routes`, each with `summary`, `steps`, optional `objective_fit`/`evidence_reaction_ids`, plus `overall_recommendation`.
- When changing molecule handling, prefer RDKit APIs over manual SMILES parsing. Existing abbreviation replacement in `clean_and_canonicalize` is intentionally small and heuristic.
- Avoid adding secrets, downloaded datasets, Qdrant storage, or virtual environments to git.
- If adding dependencies, update both local setup documentation and Docker installation instructions.

## Verification

For code changes, run syntax checks and the unit test suite:

```bash
python -m py_compile app.py scripts/index_uspto50k_to_qdrant.py src/retro_planner/*.py
python -m pytest
```

For UI changes, run Streamlit and manually verify:

- default aspirin molecule validates;
- missing Groq API key shows a warning;
- invalid structures show an error instead of calling the API;
- successful model responses render route summaries, step condition tables, text details, and either reaction images or clear warnings.
- RAG mode can fall back gracefully if Qdrant or `reaction_transforms` is unavailable.

For indexing changes, verify against a small limit before a full run:

```bash
docker compose up -d qdrant
python scripts/index_uspto50k_to_qdrant.py --limit 10 --recreate
```

The full USPTO-50K load requires network access to Hugging Face and a running Qdrant instance.

## Style

- Use plain Python with type hints where they clarify function contracts.
- Keep Streamlit labels and messages concise and practical for chemists.
- Prefer explicit error handling around external services: Groq, Hugging Face datasets, Qdrant, and RDKit parsing.
- Keep files ASCII unless there is a clear reason to preserve existing non-ASCII UI copy.
