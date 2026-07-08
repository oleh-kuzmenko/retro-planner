# 🧪 AI Retro-Synthesis Planner

An AI-powered Streamlit tool that suggests one retrosynthetic reaction for making an organic target molecule. It uses a pluggable LLM provider interface with Groq, OpenAI, and custom OpenAI-compatible implementations, RDKit for molecule validation/visualization, Ketcher for structure drawing, and optional Qdrant RAG for reaction-example context.

## Prerequisites

* **LLM Provider API Key**: Groq and OpenAI are built-in providers. The app accepts keys in the sidebar or from `GROQ_API_KEY` / `OPENAI_API_KEY`. A custom OpenAI-compatible provider can use local endpoints such as Ollama without an API key.
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
2.  **Select a provider and enter connection details**: Choose **Groq**, **OpenAI**, or **Custom / OpenAI-compatible** in the provider selector.
    * For Groq or OpenAI, paste the matching API key into the password field.
    * For a local Ollama endpoint, use base URL `http://localhost:11434/v1`, set the loaded model name, and leave the API key blank.
3.  **Adjust Retrieval and Generation (Optional)**:
    * Toggle **RAG enabled** to include or skip Qdrant retrieval.
    * Use **Top-K** to control how many retrieved reaction examples are passed as context.
    * Pick an **Optimization objective**: balanced, cheapest, fastest, or highest yield.
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
    * In RAG mode, search Qdrant for similar reactions in `reactions_morgan` and `reaction_transforms`.
    * Merge and rerank retrieved reactions with hybrid product, transformation, and reaction-class scores.
    * Send hybrid-ranked reaction context and the target structure to the selected LLM provider.
    * Request one practical forward synthesis reaction whose product is the target.

---

## 4. Interpreting Results

If the analysis is successful, you will see one proposed reaction with the following:

### 🧪 Reaction Conditions
The reaction includes reactants, product, stoichiometry, reagents, solvent, temperature, time, atmosphere, workup or purification, expected yield, and other important conditions.

### 🧠 Chemist's Reasoning
The reaction includes a brief transformation rationale and objective-fit summary.

### 🖼️ Reaction Visualizer
The app will generate a chemical equation image when RDKit can parse the returned SMILES:
`Reactants -> Product`

When RAG mode is enabled, retrieved Qdrant reactions are primarily used as model context. The app shows score metadata for transparency, but not full retrieved reaction cards.

### 🔎 Hybrid Retrieval Scores

The app also shows a score breakdown for the selected context reactions:

* **molecule_similarity**: target product Morgan fingerprint similarity.
* **reaction_similarity**: reaction transform fingerprint similarity, queried with the current target product fingerprint against the transform collection until precursor hypotheses are available.
* **reaction_class**: stored or normalized reaction class.
* **final_hybrid_score**: `0.5 * molecule_similarity + 0.3 * reaction_similarity + 0.2 * reaction_class_similarity`.

---

## ⚠️ Known Limitations

* **AI Outputs**: Verify proposed reactants, atom balance, and conditions before use.
* **Complex Stereochemistry**: The visualization tool may flatten specific stereochemical centers.
* **Single-Reaction Output**: The app requests one forward reaction to the target, not a full multi-step synthesis plan.
* **RAG Evidence**: USPTO-50K and ORD examples are useful context, but retrieved reactions may be similar rather than directly applicable.
* **Conditions Metadata**: USPTO-50K records usually lack rich experimental conditions. ORD can add solvents, temperature, catalysts, and yields when present, but model-proposed conditions should still be independently verified.

## Development Checks

For code changes, run syntax checks:

```bash
python -m py_compile app.py scripts/index_uspto50k_to_qdrant.py src/retro_planner/*.py
```

For UI changes, run Streamlit and manually verify:

* default aspirin molecule validates;
* missing provider API key shows a warning;
* invalid structures show an error instead of calling the API;
* successful model responses render one reaction, condition tables, text details, and either a reaction image or clear warnings;
* RAG mode can fall back gracefully if Qdrant or `reaction_transforms` is unavailable.

For indexing changes, run a limited indexing check:

```bash
docker compose up -d qdrant
python scripts/index_uspto50k_to_qdrant.py --limit 10 --recreate
```

The uncapped USPTO-50K + ORD load requires network access to Hugging Face and a running Qdrant instance. Use `--limit`, `--ord-allow-pattern`, or `--ord-data-dir` when you want a smaller, repeatable development run.
