import streamlit as st
import json
import os
from groq import Groq
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from streamlit_ketcher import st_ketcher
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="AI Retrosynthesis Planner", layout="wide", page_icon="🧪")

# --- Custom CSS for better UI ---
st.markdown("""
    <style>
    .stButton>button {
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧪 AI Retro-Synthesis Planner")
st.markdown("""
**Model:** Llama-3.3-70b (via Groq)  
**Instruction:** Draw a target molecule below. The AI will attempt to break it down into commercially available precursors.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", help="Required for AI generation")
    st.markdown("[Get Free Groq Key](https://console.groq.com/keys)")

    st.divider()
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.2, 0.1)
    st.info("Lower temperature (0.1-0.3) is better for strict chemistry rules.")


# --- Helper Functions ---

def clean_and_canonicalize(smiles):
    """
    Cleans custom abbreviations and returns canonical RDKit SMILES.
    """
    if not smiles: return None

    # 1. Handle common abbreviations manually if Ketcher sends them as text
    replacements = {
        "Ph": "c1ccccc1", "Et": "CC", "Me": "C",
        "Ac": "C(=O)C", "Ts": "S(=O)(=O)c1ccc(C)cc1"
    }
    s = smiles.strip()
    for k, v in replacements.items():
        # Simple string replacement (be careful with context, but effective for basics)
        if k in s: s = s.replace(k, v)

    # 2. Use RDKit to canonicalize (Fixes aromatics, standardizes format)
    try:
        mol = Chem.MolFromSmiles(s)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None


def generate_reaction_image(reactants_smiles, product_smiles):
    """
    Generates a reaction image using RDKit.
    """
    try:
        reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
        product = Chem.MolFromSmiles(product_smiles)

        if not all(reactants) or not product:
            return None

        # Create a reaction object
        rxn = AllChem.ChemicalReaction()
        for r in reactants:
            rxn.AddReactantTemplate(r)
        rxn.AddProductTemplate(product)

        # Draw
        img = Draw.ReactionToImage(rxn, subImgSize=(350, 250), useSVG=False)
        return img
    except Exception as e:
        print(f"Img Error: {e}")
        return None


def get_retrosynthesis_plan(target_smiles, client, temp):
    """
    Calls Groq Llama-3.3 to get synthesis steps.
    """
    system_prompt = """
    You are an expert Organic Chemist specializing in Retrosynthesis.
    Your task is to propose a ONE-STEP retrosynthetic disconnection for the given Target Molecule.

    CRITICAL RULES:
    1. Output MUST be valid JSON only.
    2. Use standard SMILES strings for all molecules.
    3. Verify that the reactants actually react to form the product.
    4. Do not output Markdown formatting like ```json.

    JSON STRUCTURE:
    {
      "reaction_name": "Type of reaction (e.g., Esterification, Suzuki Coupling)",
      "reactants": ["SMILES_A", "SMILES_B"],
      "reagents": "Solvents/Catalysts (e.g., H2SO4, Pd(PPh3)4)",
      "reasoning": "Brief explanation of why this disconnection works."
    }
    """

    user_prompt = f"Target Molecule SMILES: {target_smiles}"

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temp,
            response_format={"type": "json_object"}  # Forces JSON mode if model supports it
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# --- Main UI ---

# Default Acetylsalicylic acid (Aspirin)
default_smiles = "CC(=O)Oc1ccccc1C(=O)O"
smiles_input = st_ketcher(default_smiles, height=450)

# Check if input changed
canonical_input = clean_and_canonicalize(smiles_input)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Selected Target")
    if canonical_input:
        st.code(canonical_input, language="text")
        st.success("Structure Validated ✅")
    else:
        st.error("Invalid Structure or SMILES ❌")

    analyze_btn = st.button("🚀 Plan Synthesis", type="primary", use_container_width=True)

with col2:
    if analyze_btn:
        if not api_key:
            st.warning("⚠️ Please provide a Groq API Key in the sidebar.")
        elif not canonical_input:
            st.error("Please draw a valid molecule first.")
        else:
            client = Groq(api_key=api_key)

            with st.spinner("⚗️ Analyzing molecular complexity and breaking bonds..."):
                # Call AI
                result = get_retrosynthesis_plan(canonical_input, client, temperature)

                if result:
                    st.subheader(f"Proposed Path: {result.get('reaction_name', 'Unknown Reaction')}")

                    # 1. Text Details
                    st.info(f"**Reagents/Conditions:** {result.get('reagents', 'N/A')}")
                    st.markdown(f"**Chemist's Reasoning:** *{result.get('reasoning')}*")

                    # 2. Visualize Reaction
                    reactants = result.get('reactants', [])

                    # Clean AI output SMILES just in case
                    clean_reactants = [clean_and_canonicalize(r) for r in reactants]

                    # Generate Image
                    img = generate_reaction_image(clean_reactants, canonical_input)

                    if img:
                        st.image(img, caption="Retrosynthetic Step", use_container_width=False)
                    else:
                        st.warning("Could not render reaction image (SMILES might be chemically invalid).")
                        st.write("Raw Reactant SMILES:", result['reactants'])
                else:
                    st.error("Failed to generate a plan. Try a different molecule.")