# 🧪 AI Retro-Synthesis Planner

An AI-powered web tool that suggests retrosynthetic pathways for organic molecules. It uses the Groq API (Llama 3) for chemical logic and RDKit for visualization.

## Prerequisites

* **Groq API Key**: Get one for free at [console.groq.com](https://console.groq.com/keys).
* **Python 3.10+** (for local execution) or **Docker** (for containerized execution).

---

## 🚀 Option 1: Run with Docker (Recommended)

This is the easiest method as it automatically handles complex system dependencies for RDKit.

### 1. Build the Image
Navigate to the project folder in your terminal:

```bash
docker build -t retro-planner .
```

### 2. To run the Docker container you just built, execute this command in your terminal:

```bash
docker run -p 8501:8501 retro-planner
```

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
### 2. Install Dependencies
Since we are using RDKit and Streamlit, install them directly:
```bash
pip install streamlit groq rdkit streamlit-ketcher Pillow
```
Linux Users: If you see errors about libGL or libXrender, you may need to install system libraries manually: sudo apt-get install libxrender1 libxext6 libgl1

### 3. Start the App
```bash
streamlit run app.py
```

# 📝 User Guide & Usage

This application acts as an AI assistant for organic chemists, helping to brainstorm **retrosynthetic disconnections** (working backwards from a target molecule to simpler precursors).

---

## 1. Initial Setup

Before you can analyze any molecules, you must connect the application to the AI brain (Groq).

1.  **Locate the Sidebar** on the left side of the screen.
2.  **Enter your API Key**: Paste your Groq API Key into the password field.
    * *If you don't have one, click the link provided in the sidebar to generate a free key.*
3.  **Adjust Creativity (Optional)**:
    * Use the **Temperature** slider to control the AI's randomness.
    * **Low (0.1 - 0.3)**: Best for standard, well-known reactions (more reliable).
    * **High (0.7+)**: Good for exploring novel or unusual disconnections, but may produce chemically invalid results.

---

## 2. Drawing Your Target Molecule

The main interface features the **Ketcher Editor**, a professional chemical drawing tool.

* **Draw from Scratch**: Use the toolbar on the left to select atoms (C, N, O, etc.) and bonds (Single, Double, Aromatic).
* **Templates**: Use the "Benzene" or "Cyclopentadiene" icons for quick ring structures.
* **Clear/Reset**: If you make a mistake, you can clear the canvas or simply reload the page to reset to the default (Aspirin).
* **SMILES Support**: If you already have a SMILES string, you can paste it directly into the editor if the tool supports text import, or simply draw the structure corresponding to it.

> **Tip:** Ensure your valences are correct (e.g., don't draw 5 bonds to Carbon). The app attempts to "clean" the structure, but bad inputs often lead to bad AI suggestions.

---

## 3. Generating a Plan

Once your target molecule is ready:

1.  Click the red **🚀 Plan Synthesis** button.
2.  The app will:
    * Canonicalize your molecule (standardize the format).
    * Send the structure to the Llama-3 AI model.
    * Request a **one-step retrosynthetic disconnection**.

---

## 4. Interpreting Results

If the analysis is successful, you will see a result card with the following:

### ⚗️ Proposed Reaction
The AI will label the reaction type (e.g., *"Fischer Esterification"*, *"Nucleophilic Aromatic Substitution"*).

### 🧪 Reagents & Conditions
A list of necessary catalysts, solvents, or conditions required for the forward reaction (e.g., *H2SO4, Reflux, THF*).

### 🧠 Chemist's Reasoning
A brief explanation of *why* this disconnection makes sense chemically. This often includes comments on functional group compatibility or bond lability.

### 🖼️ Reaction Visualizer
The app will generate a chemical equation image showing:
`Reactants  ->  Target Molecule`

* **Left Side**: The precursors suggested by the AI.
* **Right Side**: Your original target.

---

## ⚠️ Known Limitations

* **AI Hallucinations**: The model is a text predictor, not a physics engine. Occasionally, it may suggest atoms appearing out of nowhere or disappearing. Always verify the atom count.
* **Complex Stereochemistry**: The visualization tool may flatten specific stereochemical centers (R/S) depending on the complexity.
* **Single Step**: Currently, the tool only provides the *immediate* previous step (n-1), not a full tree back to raw materials.