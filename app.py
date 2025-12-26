import streamlit as st
import json
import os
from groq import Groq
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from streamlit_ketcher import st_ketcher

st.set_page_config(page_title="Retro Synthesis Planner", layout="wide")
st.title("🧪 AI Retro-Synthesis Planner")
st.markdown("Намалюйте цільову молекулу, і AI запропонує шлях синтезу.")

with st.sidebar:
    st.header("Налаштування")
    api_key = st.text_input("Введіть Groq API Key", type="password")
    st.markdown("[Отримати ключ (безкоштовно)](https://console.groq.com/keys)")


def clean_smiles(smiles):
    if not smiles: return ""
    s = smiles.strip().replace('"', '').replace("'", "")
    replacements = {
        "Ph": "c1ccccc1", "Et": "CC", "Me": "C",
        "Ac": "C(=O)C", "Ts": "S(=O)(=O)c1ccc(C)cc1", "C6H5": "c1ccccc1"
    }
    for k, v in replacements.items():
        if k in s: s = s.replace(k, v)
    if s.endswith("Et"): s = s[:-2] + "CC"
    return s


def get_retrosynthesis(target_smiles, client):
    prompt = f"""
    Act as an expert organic chemist.
    Target molecule: {target_smiles}
    Suggest 1 synthetic pathway (Reverse synthesis).

    STRICT JSON FORMAT (No markdown):
    {{
      "steps": [
        {{
          "reactants": ["SMILES_1", "SMILES_2"],
          "product": "SMILES_PRODUCT",
          "reagents": "text description (e.g. H2SO4)"
        }}
      ]
    }}

    RULES:
    1. Output PURE SMILES only. No 'Ph', 'Et'. Use 'c1ccccc1', 'CC'.
    2. The last step's product MUST be {target_smiles}.
    3. Ensure JSON is valid.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = completion.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "{" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            content = content[start:end]
        return json.loads(content)
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None


default_smiles = "CC(=O)Oc1ccccc1C(=O)O"
smiles_input = st_ketcher(default_smiles, height=400)

col1, col2 = st.columns([1, 4])

with col1:
    analyze_btn = st.button("🚀 Знайти шлях синтезу", type="primary", use_container_width=True)

if analyze_btn:
    if not api_key:
        st.error("Будь ласка, введіть API Key у меню зліва!")
    elif not smiles_input:
        st.error("Будь ласка, намалюйте молекулу.")
    else:
        client = Groq(api_key=api_key)

        with st.spinner(f"Аналізую синтез для: {smiles_input}..."):
            data = get_retrosynthesis(smiles_input, client)

            if data and 'steps' in data:
                st.success("План побудовано!")

                for i, step in enumerate(data['steps']):
                    st.markdown(f"### Крок {i + 1}")
                    st.info(f"**Умови:** {step.get('reagents', 'Standard Conditions')}")

                    reactants_mols = []
                    for r in step['reactants']:
                        clean_r = clean_smiles(r)
                        mol = Chem.MolFromSmiles(clean_r)
                        if mol: reactants_mols.append(mol)

                    clean_p = clean_smiles(step['product'])
                    prod_mol = Chem.MolFromSmiles(clean_p)

                    if reactants_mols and prod_mol:
                        rxn_smarts = f"{'.'.join([Chem.MolToSmiles(m) for m in reactants_mols])}>>{Chem.MolToSmiles(prod_mol)}"
                        rxn = AllChem.ReactionFromSmarts(rxn_smarts)

                        img = Draw.ReactionToImage(rxn, subImgSize=(300, 200))
                        st.image(img)
                    else:
                        st.warning(f"Не вдалося намалювати крок {i + 1} (помилка в SMILES від AI)")
            else:
                st.error("Не вдалося отримати коректну відповідь від AI.")
