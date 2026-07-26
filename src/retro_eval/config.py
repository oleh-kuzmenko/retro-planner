import os
from dataclasses import dataclass


PRODUCT_COLLECTION_NAME = "reactions_morgan"
TRANSFORM_COLLECTION_NAME = "reaction_transforms"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VECTOR_SIZE = 2048


@dataclass(frozen=True)
class RetrievalWeights:
    molecule: float = 0.5
    reaction: float = 0.3
    reaction_class: float = 0.2


# Extends the PZ fig 3.2 two-component hybrid score (product + transform
# Tanimoto) with a SMARTS-heuristic reaction-class term (see
# reaction_classes.py), so retrieval can prefer precedents of the same
# disconnection type over precedents that merely look structurally similar.
DEFAULT_RETRIEVAL_WEIGHTS = RetrievalWeights()
