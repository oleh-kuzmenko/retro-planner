"""Chemistry-informed similarity groups for evaluating solvent/catalyst predictions.

Exact-SMILES comparison treats dichloromethane and chloroform (both common,
practically interchangeable chlorinated solvents) as unrelated, which is too
strict for judging whether a predicted condition is *reasonable* rather than
byte-identical. This module provides a small, hand-curated classification of
common solvents (by the standard polar-protic/polar-aprotic/nonpolar-style
families used in solvent-selection guides) and a coarser, best-effort
catalyst classification (by central metal, or by common
base/acid/organocatalyst family). Coverage is intentionally limited to
frequently-used reagents -- anything unrecognized returns None and should
fall back to exact/any-overlap SMILES comparison, not be scored as wrong.
"""

from __future__ import annotations

from retro_eval.chemistry import canonicalize_smiles

# group_name -> SMILES of representative members. Canonicalized once at
# import time so lookups are robust to equivalent-but-differently-written
# input SMILES (e.g. "C(Cl)Cl" vs "ClCCl").
_SOLVENT_GROUP_MEMBERS: dict[str, list[str]] = {
    "water": ["O"],
    "protic_alcohol": ["CO", "CCO", "CC(C)O", "CCCO", "CC(C)(C)O", "CCCCO", "OCCO"],
    "chlorinated": ["ClCCl", "ClC(Cl)Cl", "ClCCCl", "ClC(Cl)(Cl)Cl"],
    "ethereal": ["C1CCOC1", "CC1CCCO1", "O1CCOCC1", "CCOCC", "COC(C)(C)C", "CCCCOCCCC"],
    "polar_aprotic_amide_sulfoxide": ["CN(C)C=O", "CS(C)=O", "CN1CCCC1=O", "CC(=O)N(C)C", "CN(C)P(=O)(N(C)C)N(C)C"],
    "nitrile": ["CC#N"],
    "aromatic_hydrocarbon": ["Cc1ccccc1", "c1ccccc1", "Cc1ccccc1C", "Cc1cccc(C)c1", "Cc1ccc(C)cc1"],
    "ester": ["CCOC(C)=O", "COC(C)=O", "CCCCOC(C)=O"],
    "ketone": ["CC(C)=O", "CCC(C)=O", "O=C1CCCCC1"],
    "aliphatic_hydrocarbon": ["CCCCCC", "CCCCCCC", "CCCCC"],
    "amine_base_solvent": ["c1ccncc1"],
    "carboxylic_acid": ["OC=O", "CC(=O)O"],
}

_CATALYST_METAL_SYMBOLS = {
    "Pd", "Cu", "Ni", "Fe", "Ru", "Rh", "Pt", "Ir", "Co", "Mn", "Zn", "Mg", "Al", "Sn", "Ti", "Zr", "Cr", "Ag", "Au",
}

_CATALYST_GROUP_MEMBERS: dict[str, list[str]] = {
    "base_inorganic": ["[OH-].[Na+]", "[OH-].[K+]", "O=C([O-])[O-].[Na+].[Na+]", "O=C([O-])[O-].[K+].[K+]", "OC(=O)[O-].[Na+]", "OC(=O)[O-].[K+]"],
    "base_organic_amine": ["CCN(CC)CC", "CC(C)N(C(C)C)CC", "c1ccncc1", "CN(C)c1ccncc1"],
    "acid_mineral": ["Cl", "OS(=O)(=O)O"],
    "acid_organic": ["Cc1ccc(S(=O)(=O)O)cc1", "CC(=O)O", "OC(=O)C(F)(F)F"],
}


def _build_group_lookup(members: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for group, smiles_list in members.items():
        for smiles in smiles_list:
            canonical = canonicalize_smiles(smiles)
            if canonical:
                lookup[canonical] = group
    return lookup


_SOLVENT_LOOKUP = _build_group_lookup(_SOLVENT_GROUP_MEMBERS)
_CATALYST_LOOKUP = _build_group_lookup(_CATALYST_GROUP_MEMBERS)


def solvent_group(value: str | None) -> str | None:
    """Best-effort chemical family for one solvent component, or None if unrecognized."""
    if not value:
        return None
    canonical = canonicalize_smiles(value)
    if canonical and canonical in _SOLVENT_LOOKUP:
        return _SOLVENT_LOOKUP[canonical]
    return None


def catalyst_group(value: str | None) -> str | None:
    """Best-effort chemical family for one catalyst/reagent component, or None if unrecognized.

    Tries the small curated base/acid table first, then falls back to
    classifying by central metal (RDKit atom symbols) for organometallic
    catalysts not explicitly listed -- coarse but covers the common
    transition-metal-catalyzed reaction families (Pd/Cu/Ni-catalyzed
    couplings etc.) without needing an exhaustive reagent database.
    """
    if not value:
        return None
    canonical = canonicalize_smiles(value)
    if canonical and canonical in _CATALYST_LOOKUP:
        return _CATALYST_LOOKUP[canonical]

    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    metals = symbols & _CATALYST_METAL_SYMBOLS
    if metals:
        return f"metal_{sorted(metals)[0]}"
    return None


def component_groups(value: str | None, classifier) -> frozenset[str] | None:
    """Apply `classifier` to every "."/","-separated component of `value`.

    Returns None if there are no recognizable components at all (so callers
    can fall back to exact/any-overlap SMILES comparison instead of
    penalizing an unrecognized-but-possibly-correct answer).
    """
    if not value or value == "not specified":
        return None
    parts = [p.strip() for p in str(value).replace(",", ".").split(".") if p.strip() and p.strip() != "not specified"]
    groups = {classifier(part) for part in parts}
    groups.discard(None)
    return frozenset(groups) if groups else None
