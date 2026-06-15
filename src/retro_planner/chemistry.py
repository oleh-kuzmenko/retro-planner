from typing import Optional

import numpy as np
from rdkit import Chem
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


def canonicalize_smiles(smiles: str | None) -> Optional[str]:
    """Return canonical RDKit SMILES after a small Ketcher abbreviation cleanup."""
    if not smiles:
        return None

    normalized = str(smiles).strip()
    for abbreviation, replacement in SMILES_ABBREVIATIONS.items():
        normalized = normalized.replace(abbreviation, replacement)

    try:
        mol = Chem.MolFromSmiles(normalized)
    except Exception:
        return None

    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def clean_and_canonicalize(smiles: str | None) -> Optional[str]:
    return canonicalize_smiles(smiles)


def morgan_array(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
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


def mol_from_smiles_without_atom_maps(smiles: str | None):
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
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
