import os
from dataclasses import dataclass


PRODUCT_COLLECTION_NAME = "reactions_morgan"
TRANSFORM_COLLECTION_NAME = "reaction_transforms"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VECTOR_SIZE = 2048
DEFAULT_TARGET_SMILES = "O1C(C(=O)OC)=CC=C1S(F)(=O)=O"


@dataclass(frozen=True)
class RetrievalWeights:
    molecule: float = 0.5
    reaction: float = 0.5
    reaction_class: float = 0.0


# PZ fig 3.2: the hybrid score is exactly two Tanimoto components (product +
# transform). reaction_class defaults to 0.0 so it does not contribute.
DEFAULT_RETRIEVAL_WEIGHTS = RetrievalWeights()

# Extension beyond the PZ architecture: blends in a SMARTS-heuristic
# reaction-class similarity (see reaction_classes.py). Opt in explicitly via
# `RetrievalConfig(weights=EXPERIMENTAL_RETRIEVAL_WEIGHTS)`; not used by default.
EXPERIMENTAL_RETRIEVAL_WEIGHTS = RetrievalWeights(
    molecule=0.5,
    reaction=0.3,
    reaction_class=0.2,
)
