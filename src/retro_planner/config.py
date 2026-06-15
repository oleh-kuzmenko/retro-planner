import os
from dataclasses import dataclass


PRODUCT_COLLECTION_NAME = "reactions_morgan"
TRANSFORM_COLLECTION_NAME = "reaction_transforms"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VECTOR_SIZE = 2048
DEFAULT_TARGET_SMILES = "O1C(C(=O)OC)=CC=C1S(F)(=O)=O"
OPTIMIZATION_OBJECTIVES = ("BALANCED", "CHEAPEST", "FASTEST", "HIGHEST_YIELD")


@dataclass(frozen=True)
class RetrievalWeights:
    molecule: float = 0.5
    reaction: float = 0.3
    reaction_class: float = 0.2


DEFAULT_RETRIEVAL_WEIGHTS = RetrievalWeights()
