from rdkit import Chem


CLASS_ALIASES = {
    "1": "alkylation_arylation",
    "2": "acylation",
    "3": "carbon_carbon_coupling",
    "4": "heterocycle_formation",
    "5": "protection",
    "6": "deprotection",
    "7": "reduction",
    "8": "oxidation",
    "9": "functional_group_interconversion",
    "10": "functional_group_addition",
}

CLASS_SIMILARITY_GROUPS = [
    {"acylation", "esterification", "amidation"},
    {"oxidation", "reduction"},
    {"protection", "deprotection"},
    {"coupling", "carbon_carbon_coupling", "suzuki", "heck", "sonogashira"},
    {"functional_group_interconversion", "functional_group_addition"},
]


def normalize_reaction_class(reaction_class) -> str | None:
    if reaction_class is None:
        return None

    label = str(reaction_class).strip().lower()
    if not label:
        return None

    if label in CLASS_ALIASES:
        return CLASS_ALIASES[label]

    label = label.replace("-", " ").replace("_", " ")
    keyword_map = {
        "acyl": "acylation",
        "ester": "esterification",
        "amid": "amidation",
        "oxid": "oxidation",
        "reduc": "reduction",
        "deprotect": "deprotection",
        "protect": "protection",
        "coupling": "coupling",
        "suzuki": "suzuki",
        "heck": "heck",
        "sonogashira": "sonogashira",
        "heterocycle": "heterocycle_formation",
        "alkyl": "alkylation_arylation",
        "aryl": "alkylation_arylation",
        "c-c": "carbon_carbon_coupling",
        "carbon carbon": "carbon_carbon_coupling",
        "functional group interconversion": "functional_group_interconversion",
        "functional group addition": "functional_group_addition",
    }
    for keyword, normalized in keyword_map.items():
        if keyword in label:
            return normalized

    return label.replace(" ", "_")


def infer_target_reaction_classes(target_smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return set()

    inferred: set[str] = set()
    smarts_to_class = {
        "[CX3](=O)[OX2H0]": "acylation",
        "[CX3](=O)[NX3]": "amidation",
        "[OX2][CX4]": "protection",
        "[NX3][CX4]": "alkylation_arylation",
        "c[Br,Cl,I]": "coupling",
        "[CX3]=[OX1]": "oxidation",
    }
    for smarts, reaction_class in smarts_to_class.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            inferred.add(reaction_class)

    return inferred


def reaction_class_similarity(candidate_class, target_classes: set[str]) -> float:
    normalized = normalize_reaction_class(candidate_class)
    if not normalized or not target_classes:
        return 0.0
    if normalized in target_classes:
        return 1.0

    for group in CLASS_SIMILARITY_GROUPS:
        if normalized in group and target_classes.intersection(group):
            return 0.5
    return 0.0
