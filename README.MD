# 🧪 AI Retro-Synthesis Planner

An AI-powered Streamlit tool that suggests retrosynthetic route options for organic molecules. It uses a pluggable LLM provider interface with Groq-hosted Llama models as the current implementation, RDKit for molecule validation/visualization, Ketcher for structure drawing, and optional Qdrant RAG for reaction-example context.

## Prerequisites

* **LLM Provider API Key**: Groq is the current provider. Get a key at [console.groq.com](https://console.groq.com/keys). The app accepts it in the sidebar or from `GROQ_API_KEY`.
* **Python 3.10+** for local execution, or **Docker** for containerized app execution.
* **Qdrant** for RAG mode. The included `docker-compose.yml` starts local Qdrant on ports `6333` and `6334`.

Useful environment variables:

```bash
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
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

## 🐍 Option 2: Run Locally (Python)

If you prefer running without Docker, ensure you have Python 3.10+ installed.

### 1. Create a Virtual Environment
It is best practice to use a virtual environment to avoid conflicts.

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

Install the test dependency for local development:

```bash
pip install -e ".[dev]"
```

To install both development and indexing extras in one step:

```bash
pip install -e ".[dev,indexing]"
```

### 3. Start Qdrant

```bash
docker compose up -d qdrant
```

### 4. Index Reaction Examples for RAG

The indexer rebuilds the Qdrant RAG database from scratch. It drops and recreates both vector collections before loading data, so run it only when you are ready to replace the existing local index.

For a small smoke test against both default sources:

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

# 📝 User Guide & Usage

This application acts as an AI assistant for organic chemists, helping to brainstorm **retrosynthetic disconnections** (working backwards from a target molecule to simpler precursors).

---

## 1. Initial Setup

Before you can analyze any molecules, you must connect the application to an LLM provider. Groq is the current built-in provider.

1.  **Locate the Sidebar** on the left side of the screen.
2.  **Select a provider and enter your API key**: Choose **Groq** in the provider selector and paste your Groq API key into the password field.
    * *If you don't have one, click the link provided in the sidebar to generate a free key.*
3.  **Adjust Retrieval and Generation (Optional)**:
    * Toggle **RAG enabled** to include or skip Qdrant retrieval.
    * Use **Top-K** to control how many retrieved reaction examples are passed as context.
    * Use **Route options** to request 1-5 distinct routes.
    * Pick an **Optimization objective**: balanced, cheapest, fastest, or highest yield.
    * Edit the provider model field if you want to use a different compatible model.
4.  **Adjust Creativity (Optional)**:
    * Use the **Temperature** slider to control the AI's randomness.
    * **Low (0.1 - 0.3)**: Best for standard, well-known reactions (more reliable).
    * **High (0.7+)**: Good for exploring novel or unusual disconnections, but may produce chemically invalid results.

---

## 2. Entering Your Target Molecule

The main interface supports both the **Ketcher Editor** and direct SMILES entry.

* **Draw from Scratch**: Use the toolbar on the left to select atoms (C, N, O, etc.) and bonds (Single, Double, Aromatic).
* **Templates**: Use the "Benzene" or "Cyclopentadiene" icons for quick ring structures.
* **Clear/Reset**: If you make a mistake, you can clear the canvas or simply reload the page to reset to the default (Aspirin).
* **SMILES Support**: Choose **SMILES** as the input source and paste a SMILES string into the text field.

> **Tip:** Ensure your valences are correct (e.g., don't draw 5 bonds to Carbon). The app attempts to "clean" the structure, but bad inputs often lead to bad AI suggestions.

---

## 3. Generating a Plan

Once your target molecule is ready:

1.  Click the red **Generate retrosynthesis** button.
2.  The app will:
    * Canonicalize your molecule (standardize the format).
    * In RAG mode, search Qdrant for similar reactions in `reactions_morgan` and `reaction_transforms`.
    * Merge and rerank retrieved reactions with hybrid product, transformation, and reaction-class scores.
    * Send hybrid-ranked reaction context and the target structure to the selected LLM provider.
    * Request several route options, each with 1-5 forward synthesis steps.

---

## 4. Interpreting Results

If the analysis is successful, you will see route options with the following:

### ⚗️ Route Summaries
Each route includes a short strategy plus difficulty, cleanliness, cost, expected overall yield, major risks, and best use case.

### 🧪 Step Conditions
Each step includes reactants, product, stoichiometry, reagents, solvent, temperature, time, atmosphere, workup or purification, expected yield, and other important conditions.

### 🧠 Chemist's Reasoning
Each step includes a brief explanation of why the transformation is plausible. Each route also notes where it is cheapest, cleanest, easiest, or most useful.

### 🖼️ Reaction Visualizer
The app will generate chemical equation images for steps where RDKit can parse the returned SMILES:
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

* **AI Hallucinations**: The model is a text predictor, not a physics engine. Occasionally, it may suggest atoms appearing out of nowhere or disappearing. Always verify the atom count.
* **Complex Stereochemistry**: The visualization tool may flatten specific stereochemical centers (R/S) depending on the complexity.
* **Route Depth**: Routes are limited to 1-5 linear steps, not a full branching synthesis tree.
* **RAG Evidence**: USPTO-50K and ORD examples are useful context, but retrieved reactions may be similar rather than directly applicable.
* **Conditions Metadata**: USPTO-50K records usually lack rich experimental conditions. ORD can add solvents, temperature, catalysts, and yields when present, but model-proposed conditions should still be independently verified.

## Development Checks

For code changes, run syntax checks and the unit test suite:

```bash
python -m py_compile app.py scripts/index_uspto50k_to_qdrant.py src/retro_planner/*.py
python -m pytest
```

For indexing changes, smoke test against a small limit first:

```bash
docker compose up -d qdrant
python scripts/index_uspto50k_to_qdrant.py --limit 10 --recreate
```

The uncapped USPTO-50K + ORD load requires network access to Hugging Face and a running Qdrant instance. Use `--limit`, `--ord-allow-pattern`, or `--ord-data-dir` when you want a smaller, repeatable development run.

`testing/test-reactions` contains manual golden-product prompts and scoring notes for comparing no-RAG, product-RAG, and hybrid-RAG behavior.
