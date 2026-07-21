# 🧪 AI Retro-Synthesis Planner

An AI-powered Streamlit tool that performs a single-step chain-of-thought retrosynthetic analysis for an organic target molecule: one call to one LLM proposes one practical bond disconnection, returned as plain `<think>...</think>` reasoning plus `<answer>...</answer>` reactant SMILES (no JSON). It uses a pluggable `LLMProvider` registry — Groq, OpenAI, and custom OpenAI-compatible chat APIs, plus optional local/research backends (ReactionT5v2 seq2seq, a two-stage fine-tuned Qwen2.5 LoRA, ChemLLM GGUF) — RDKit for molecule validation/visualization, Ketcher for structure drawing, and optional Qdrant RAG for reaction-example context. Additional candidate disconnections for the same target come from repeating the single-step call, not from asking the model for multiple routes at once.

## Prerequisites

* **LLM Provider API Key**: Groq and OpenAI are built-in cloud providers. The app accepts keys in the sidebar or from `GROQ_API_KEY` / `OPENAI_API_KEY`. A custom OpenAI-compatible provider can use local endpoints such as Ollama without an API key. Local/research providers (ReactionT5v2, the Qwen LoRA pair, ChemLLM GGUF) need no API key but require `pip install -e ".[local-models]"` (torch, transformers, peft, llama-cpp-python) and download model weights on first use.
* **Python 3.10+** for local execution, or **Docker** for containerized app execution.
* **Qdrant** for RAG mode. The included `docker-compose.yml` starts local Qdrant on ports `6333` and `6334`.

Useful environment variables:

```bash
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1
CUSTOM_LLM_BASE_URL=http://localhost:11434/v1
CUSTOM_LLM_MODEL=llama3.1
CUSTOM_LLM_API_KEY=...
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

## 🚀 Option 1: Run the App with Docker

This image runs the Streamlit app and includes the app dependencies. Start Qdrant separately if you want RAG mode.

### 1. Build the Image
Navigate to the project folder in your terminal:

```bash
docker build -t retro-planner .
```

### 2. Run Without RAG

```bash
docker run -p 8501:8501 retro-planner
```

Open http://localhost:8501 and turn off **RAG enabled** in the sidebar if Qdrant is not running.

### 3. Run With Local Qdrant

Start Qdrant:

```bash
docker compose up -d qdrant
```

On macOS/Windows Docker Desktop, point the app container at the host Qdrant port:

```bash
docker run -p 8501:8501 -e QDRANT_HOST=host.docker.internal retro-planner
```

On Linux, use your Docker network or host networking equivalent so the app can reach Qdrant.

For a local Ollama model from inside the app container on macOS/Windows Docker Desktop, set the custom provider base URL to `http://host.docker.internal:11434/v1`.

## 🐍 Option 2: Run Locally (Python)

If you prefer running without Docker, ensure you have Python 3.10+ installed.

### 1. Create a Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
```
### 2. Install App Dependencies

```bash
pip install -e .
```

Linux users: if RDKit image rendering fails because of missing GUI libraries, install system libraries such as:

```bash
sudo apt-get install libxrender1 libxext6 libgl1
```

Install the extra indexing dependencies if you plan to load USPTO-50K and ORD reactions into Qdrant:

```bash
pip install -e ".[indexing]"
```

### 3. Start Qdrant

```bash
docker compose up -d qdrant
```

### 4. Index Reaction Examples for RAG

The indexer rebuilds the Qdrant RAG database from scratch. It drops and recreates both vector collections before loading data.

Index a limited sample from both default sources:

```bash
python scripts/index_uspto50k_to_qdrant.py --limit 100 --recreate
```

By default, the configured USPTO-50K + ORD run indexes up to 500,000 reactions total:

```bash
python scripts/index_uspto50k_to_qdrant.py --recreate
```

To process all available reactions instead of using the default cap:

```bash
python scripts/index_uspto50k_to_qdrant.py --limit 0 --recreate
```

The `--recreate` flag is still accepted for older commands, but collections are now always recreated. The indexer downloads `pingzhili/uspto-50k` through `datasets` and, unless `--ord-data-dir` is provided, downloads ORD protobuf files from the Hugging Face mirror `Open-Reaction-Database/ord-data`.

To index only one source:

```bash
python scripts/index_uspto50k_to_qdrant.py --sources uspto --limit 100
python scripts/index_uspto50k_to_qdrant.py --sources ord --limit 100
```

To use a local ORD checkout or a single ORD protobuf file:

```bash
python scripts/index_uspto50k_to_qdrant.py --sources ord --ord-data-dir /path/to/ord-data
python scripts/index_uspto50k_to_qdrant.py --sources ord --ord-data-dir /path/to/sample.pb.gz
```

To download only a subset of ORD from Hugging Face:

```bash
python scripts/index_uspto50k_to_qdrant.py --sources ord --ord-allow-pattern "data/4d/*.pb.gz"
```

The indexer writes a shared payload schema with `product_smiles`, `reactant_smiles`, `reactants_smiles`, `conditions`, and `source`. USPTO records have empty/unknown conditions. ORD records attempt to include solvents, temperature in Celsius, catalysts, and yield when present in the source protobuf.

It creates two Qdrant collections:

* `reactions_morgan`: product Morgan fingerprints for molecule similarity.
* `reaction_transforms`: MVP reaction fingerprints computed as `product_fp XOR reactant_fp`.

### 5. Start the App

```bash
streamlit run app.py
```

### 6. Use a Local Ollama Model

Ollama exposes an OpenAI-compatible API at `/v1`.

```bash
ollama serve
ollama pull llama3.1
```

In the sidebar, choose **Custom / OpenAI-compatible**, set the base URL to `http://localhost:11434/v1`, set the model to `llama3.1`, and leave the API key blank.

# 📝 User Guide & Usage

This application acts as an AI assistant for organic chemists, helping to brainstorm **retrosynthetic disconnections** (working backwards from a target molecule to simpler precursors).

---

## 1. Initial Setup

Before you can analyze any molecules, you must connect the application to an LLM provider.

1.  **Locate the Sidebar** on the left side of the screen.
2.  **Pick a provider category**: **Cloud API** (Groq, OpenAI, Custom / OpenAI-compatible) or **Local / research** (ReactionT5v2, two-stage Qwen LoRA, ChemLLM GGUF). The local category shows a warning about the extra dependencies it needs.
3.  **Select a provider and enter connection details**:
    * For Groq or OpenAI, paste the matching API key into the password field.
    * For a local Ollama endpoint, use base URL `http://localhost:11434/v1`, set the loaded model name, and leave the API key blank.
    * For local/research providers, no API key is required; the first generation call loads model weights, which can take a while.
4.  **Adjust Retrieval and Generation (Optional)**:
    * Toggle **RAG enabled** to include or skip Qdrant retrieval.
    * Use **Top-K** to control how many retrieved reaction examples are passed as context.
    * Edit the provider model field if you want to use a different compatible model.
---

## 2. Entering Your Target Molecule

The main interface supports both the **Ketcher Editor** and direct SMILES entry.

* **Draw from Scratch**: Use the toolbar on the left to select atoms (C, N, O, etc.) and bonds (Single, Double, Aromatic).
* **Templates**: Use the "Benzene" or "Cyclopentadiene" icons for quick ring structures.
* **Clear/Reset**: Clear the canvas or reload the page to reset to the default molecule.
* **SMILES Support**: Choose **SMILES** as the input source and paste a SMILES string into the text field.

---

## 3. Generating a Plan

Once your target molecule is ready:

1.  Click the red **Generate retrosynthesis** button.
2.  The app will:
    * Canonicalize your molecule (standardize the format).
    * In RAG mode, search Qdrant for similar reactions in `reactions_morgan` and `reaction_transforms`, then rerank the shortlist by an exact Tanimoto coefficient (`0.5 * product similarity + 0.5 * transform similarity` by default).
    * Build a 4-block `[System]/[Context]/[Instruction]/[Input]` chain-of-thought prompt from the retrieved examples and the target structure, and send it to the selected LLM provider for a single retrosynthetic step.
    * Parse the `<think>`/`<answer>` response, chemically validate the reactant SMILES with RDKit, and automatically retry once with a repair prompt if the answer was missing or invalid.
3.  Click **🔁 Generate another candidate** to get an additional single-step disconnection for the same target, reusing the already-retrieved RAG context instead of querying Qdrant again.

---

## 4. Interpreting Results

If the analysis is successful, you will see one proposed retrosynthetic step with the following:

### 🧠 Chemist's Reasoning
An expander with the model's raw `<think>` (or `<reason>`) text: step-by-step analysis of functional groups, likely reaction centers, and the feasibility of the proposed disconnection. This is shown verbatim so a chemist can verify the model's logic, not just its answer.

### Proposed Single-Step Disconnection
The precursor SMILES parsed out of `<answer>` (dot-separated, RDKit-validated) next to the canonical target/product SMILES.

### 🖼️ Reaction Visualizer
The app will generate a chemical equation image when RDKit can parse the returned SMILES:
`Reactants -> Product`

If the model's answer fails validation (unparseable SMILES, or a mass balance far below the target), the app automatically retries once with a repair prompt using the same tag contract before giving up and showing the warning/error instead.

When RAG mode is enabled, retrieved Qdrant reactions are primarily used as model context. The app shows score metadata for transparency, but not full retrieved reaction cards.

### 🔎 Hybrid Retrieval Scores

The app also shows a score breakdown for the selected context reactions:

* **molecule_similarity**: exact Tanimoto coefficient between the target and the retrieved product's Morgan fingerprint.
* **reaction_similarity**: exact Tanimoto coefficient between the target's reaction-transform fingerprint and the retrieved transform (`product_fp XOR reactant_fp`).
* **reaction_class_similarity**: SMARTS-heuristic reaction-class match. This is an extension beyond the source thesis architecture and defaults to weight `0.0` (see `EXPERIMENTAL_RETRIEVAL_WEIGHTS` in `config.py` to opt in).
* **final_hybrid_score**: `0.5 * molecule_similarity + 0.5 * reaction_similarity` by default (the two-component formula from the target architecture); `0.5 / 0.3 / 0.2` across all three scores when the experimental weights are enabled.

---

## ⚠️ Known Limitations

* **AI Outputs**: Verify proposed reactants and atom balance before use; the app only checks RDKit validity and a rough heavy-atom mass balance, not reaction feasibility.
* **No Reaction Conditions**: The model returns only precursor SMILES, not stoichiometry, solvent, temperature, or yield. Retrieved RAG examples may include conditions metadata (from ORD), but that is retrieval context, not a model-proposed condition set.
* **Complex Stereochemistry**: The visualization tool may flatten specific stereochemical centers.
* **Single-Step Output**: Each generation call proposes one retrosynthetic step, not a full multi-step synthesis plan. Use **Generate another candidate** for alternative single-step disconnections of the same target.
* **RAG Evidence**: USPTO-50K and ORD examples are useful context, but retrieved reactions may be similar rather than directly applicable.
* **Legacy No-Think Providers**: Seq2seq providers such as ReactionT5v2 don't produce a `<think>` block; the app shows no "Chemist's Reasoning" expander for those responses, which is expected, not a bug.

## Development Checks

For code changes, run syntax checks and the test suite:

```bash
python -m py_compile app.py scripts/index_uspto50k_to_qdrant.py src/retro_planner/*.py src/retro_planner/providers/*.py
pip install -e ".[test]"
python -m pytest tests/
```

The test suite (`tests/`) covers prompt construction, response parsing/validation, retrieval scoring, the provider registry, and evaluation metrics without needing Qdrant, network access, or GPU/local-model dependencies.

For UI changes, run Streamlit and manually verify:

* default aspirin molecule validates;
* missing provider API key (or base URL for the custom provider) shows a warning instead of calling the API;
* invalid structures show an error instead of calling the API;
* successful model responses render the "Chemist's Reasoning" expander, the precursors/product codes, and either a reaction image or a clear warning;
* "Generate another candidate" adds another step without re-querying Qdrant;
* RAG mode can fall back gracefully if Qdrant or `reaction_transforms` is unavailable.

To reproduce the Table 4.1-style automated evaluation (Top-1/3/5 exact match, Structure Success Rate) on a USPTO-50K sample:

```bash
pip install -e ".[indexing]"
GROQ_API_KEY=... python scripts/evaluate_retrosynthesis.py --limit 25
```

For indexing changes, run a limited indexing check:

```bash
docker compose up -d qdrant
python scripts/index_uspto50k_to_qdrant.py --limit 10 --recreate
```

The uncapped USPTO-50K + ORD load requires network access to Hugging Face and a running Qdrant instance. Use `--limit`, `--ord-allow-pattern`, or `--ord-data-dir` when you want a smaller, repeatable development run.
