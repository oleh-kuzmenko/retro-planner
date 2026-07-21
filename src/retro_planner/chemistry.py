from typing import Optional

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

from retro_planner.config import VECTOR_SIZE


MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=VECTOR_SIZE,
)

SMILES_ABBREVIATIONS = {
    "Ph": "c1ccccc1",
    "Et": "CC",
    "Me": "C",
    "Ac": "C(=O)C",
    "Ts": "S(=O)(=O)c1ccc(C)cc1",
}

INORGANIC_FORMULA_SMILES = {
    "Br2": "[Br][Br]",
    "Cl2": "[Cl][Cl]",
    "F2": "[F][F]",
    "H2": "[H][H]",
    "I2": "[I][I]",
    "KOH": "[K+].[OH-]",
    "LiOH": "[Li+].[OH-]",
    "NaOH": "[Na+].[OH-]",
    "FS(O)(=O)Na": "[Na+].[O-]S(=O)(=O)F",
    "NaSO2F": "[Na+].[O-]S(=O)(=O)F",
    "PPh3": "P(c1ccccc1)(c1ccccc1)c1ccccc1",
    "Pd(PPh3)4": (
        "[Pd]."
        "P(c1ccccc1)(c1ccccc1)c1ccccc1."
        "P(c1ccccc1)(c1ccccc1)c1ccccc1."
        "P(c1ccccc1)(c1ccccc1)c1ccccc1."
        "P(c1ccccc1)(c1ccccc1)c1ccccc1"
    ),
}


def normalize_formula_smiles(smiles: str | None) -> str | None:
    """Normalize supported formula-style reagent names to parseable SMILES."""
    if not smiles:
        return None

    normalized = str(smiles).strip()
    return INORGANIC_FORMULA_SMILES.get(normalized, normalized)


def is_known_formula_smiles(smiles: str | None) -> bool:
    if not smiles:
        return False
    return str(smiles).strip() in INORGANIC_FORMULA_SMILES


def canonicalize_smiles(smiles: str | None) -> Optional[str]:
    """Return canonical RDKit SMILES after a small Ketcher abbreviation cleanup."""
    if not smiles:
        return None

    normalized = normalize_formula_smiles(smiles)
    if not normalized:
        return None
    for abbreviation, replacement in SMILES_ABBREVIATIONS.items():
        normalized = normalized.replace(abbreviation, replacement)

    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(normalized)
    except Exception:
        return None

    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def clean_and_canonicalize(smiles: str | None) -> Optional[str]:
    return canonicalize_smiles(smiles)


def morgan_array(smiles: str) -> Optional[np.ndarray]:
    normalized = normalize_formula_smiles(smiles)
    if not normalized:
        return None

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        return None

    fingerprint = MORGAN_GENERATOR.GetFingerprint(mol)
    array = np.zeros((VECTOR_SIZE,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def generate_morgan_fingerprint_array(smiles: str) -> np.ndarray:
    array = morgan_array(smiles)
    if array is None:
        raise ValueError("Invalid SMILES cannot be fingerprinted.")
    return array


def morgan_vector(smiles: str) -> Optional[list[float]]:
    array = morgan_array(smiles)
    if array is None:
        return None
    return array.tolist()


def generate_morgan_fingerprint(smiles: str) -> list[float]:
    return generate_morgan_fingerprint_array(smiles).tolist()


def _split_reactants(reactants_smiles) -> list[str]:
    if isinstance(reactants_smiles, str):
        return [part for part in reactants_smiles.split(".") if part]
    if isinstance(reactants_smiles, list):
        return [str(part) for part in reactants_smiles if part]
    return []


def combined_reactant_array(reactants_smiles) -> np.ndarray:
    combined = np.zeros((VECTOR_SIZE,), dtype=np.float32)
    for smiles in _split_reactants(reactants_smiles):
        array = morgan_array(smiles)
        if array is not None:
            combined = np.maximum(combined, array)
    return combined


def combined_reactant_fingerprint_array(reactants_smiles) -> np.ndarray:
    return combined_reactant_array(reactants_smiles)


def reaction_transform_vector(
    product_smiles: str,
    reactants_smiles=None,
) -> Optional[list[float]]:
    product_fingerprint = morgan_array(product_smiles)
    if product_fingerprint is None:
        return None

    reactant_fingerprint = combined_reactant_array(reactants_smiles)
    transform = np.logical_xor(
        product_fingerprint.astype(bool),
        reactant_fingerprint.astype(bool),
    )
    return transform.astype(np.float32).tolist()


def generate_reaction_fingerprint(
    product_smiles: str,
    reactants_smiles=None,
) -> list[float]:
    vector = reaction_transform_vector(product_smiles, reactants_smiles)
    if vector is None:
        raise ValueError("Invalid product SMILES cannot be fingerprinted.")
    return vector


def tanimoto_similarity(vector_a, vector_b) -> float:
    """Bitwise Tanimoto similarity between two binary fingerprint vectors.

    Both fingerprints produced by this module (Morgan and the XOR-based
    reaction transform) are 0/1 vectors, so Tanimoto reduces to
    popcount(a & b) / popcount(a | b) and can be computed directly on the
    raw vectors without going through an RDKit ExplicitBitVect.
    """
    a = np.asarray(vector_a, dtype=bool)
    b = np.asarray(vector_b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    return float(intersection) / float(union)


def mol_from_smiles_without_atom_maps(smiles: str | None):
    if not smiles:
        return None

    normalized = normalize_formula_smiles(smiles)
    if not normalized:
        return None

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        return None

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def parse_reaction_smiles(reaction_smiles: str | None):
    if not reaction_smiles or ">>" not in reaction_smiles:
        return None

    reactants_part, products_part = reaction_smiles.split(">>", maxsplit=1)
    reactants = [part for part in reactants_part.split(".") if part]
    products = [part for part in products_part.split(".") if part]

    if not reactants or not products:
        return None
    return reactants, products
