from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from retro_planner.chemistry import (
    generate_morgan_fingerprint,
    generate_reaction_fingerprint,
)
from retro_planner.config import (
    DEFAULT_RETRIEVAL_WEIGHTS,
    PRODUCT_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    TRANSFORM_COLLECTION_NAME,
    RetrievalWeights,
)
from retro_planner.reaction_classes import (
    infer_target_reaction_classes,
    reaction_class_similarity,
)

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


@dataclass(frozen=True)
class RetrievalConfig:
    product_collection: str = PRODUCT_COLLECTION_NAME
    transform_collection: str = TRANSFORM_COLLECTION_NAME
    host: str = QDRANT_HOST
    port: int = QDRANT_PORT
    timeout: int = 10
    weights: RetrievalWeights = DEFAULT_RETRIEVAL_WEIGHTS


@dataclass(frozen=True)
class RetrievalResult:
    reactions: list[dict]
    warnings: list[str]


def create_qdrant_client(config: RetrievalConfig | None = None) -> QdrantClient:
    from qdrant_client import QdrantClient

    resolved = config or RetrievalConfig()
    client = QdrantClient(
        host=resolved.host,
        port=resolved.port,
        timeout=resolved.timeout,
    )
    client.get_collections()
    return client


def _payload_value(payload: dict, key: str):
    value = payload.get(key)
    return value if value is not None else None


def query_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector: list[float],
    top_k: int,
):
    try:
        return client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
    except AttributeError:
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return response.points


def reaction_from_hit(hit, score_key: str) -> dict:
    payload = hit.payload or {}
    reaction = {
        "reaction_id": _payload_value(payload, "reaction_id"),
        "reactants_smiles": _payload_value(payload, "reactants_smiles"),
        "product_smiles": _payload_value(payload, "product_smiles"),
        "reaction_smiles": _payload_value(payload, "reaction_smiles"),
        "reaction_class": _payload_value(payload, "reaction_class"),
        "reaction_class_normalized": _payload_value(
            payload,
            "reaction_class_normalized",
        ),
        "solvent": _payload_value(payload, "solvent"),
        "temperature_celsius": _payload_value(payload, "temperature_celsius"),
        "pressure_atm": _payload_value(payload, "pressure_atm"),
        "reaction_time_hours": _payload_value(payload, "reaction_time_hours"),
        "yield_percent": _payload_value(payload, "yield_percent"),
        "source": _payload_value(payload, "source"),
    }
    reaction[score_key] = float(hit.score)
    return reaction


def search_similar_reactions(
    vector: list[float],
    top_k: int = 10,
    client: QdrantClient | None = None,
    config: RetrievalConfig | None = None,
) -> list[dict]:
    resolved = config or RetrievalConfig()
    qdrant = client or create_qdrant_client(resolved)
    hits = query_qdrant_collection(
        qdrant,
        resolved.product_collection,
        vector,
        top_k,
    )
    return [reaction_from_hit(hit, "molecule_similarity") for hit in hits]


def merge_retrieval_hits(
    molecule_hits: list[dict],
    transform_hits: list[dict],
    target_classes: set[str],
    weights: RetrievalWeights = DEFAULT_RETRIEVAL_WEIGHTS,
) -> list[dict]:
    merged: dict[str, dict] = {}

    for reaction in molecule_hits + transform_hits:
        reaction_id = reaction.get("reaction_id") or reaction.get("reaction_smiles")
        if not reaction_id:
            continue

        current = merged.setdefault(str(reaction_id), {})
        for key, value in reaction.items():
            if value is None:
                continue
            if key in {"molecule_similarity", "reaction_similarity"}:
                current[key] = max(float(value), float(current.get(key, 0.0)))
            else:
                current.setdefault(key, value)

    reranked = []
    for reaction in merged.values():
        molecule_similarity = float(reaction.get("molecule_similarity", 0.0))
        reaction_similarity = float(reaction.get("reaction_similarity", 0.0))
        class_source = (
            reaction.get("reaction_class_normalized")
            or reaction.get("reaction_class")
        )
        class_similarity = reaction_class_similarity(class_source, target_classes)
        final_score = (
            weights.molecule * molecule_similarity
            + weights.reaction * reaction_similarity
            + weights.reaction_class * class_similarity
        )
        reaction["reaction_class_similarity"] = class_similarity
        reaction["final_hybrid_score"] = final_score
        reranked.append(reaction)

    return sorted(
        reranked,
        key=lambda reaction: reaction.get("final_hybrid_score", 0.0),
        reverse=True,
    )


def hybrid_retrieve_reactions_for_smiles(
    smiles: str,
    top_k: int,
    client: QdrantClient | None = None,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    resolved = config or RetrievalConfig()
    qdrant = client or create_qdrant_client(resolved)
    product_vector = generate_morgan_fingerprint(smiles)
    transform_vector = generate_reaction_fingerprint(smiles)
    query_limit = max(top_k * 3, top_k)
    warnings: list[str] = []

    molecule_hits = [
        reaction_from_hit(hit, "molecule_similarity")
        for hit in query_qdrant_collection(
            qdrant,
            resolved.product_collection,
            product_vector,
            query_limit,
        )
    ]

    try:
        transform_hits = [
            reaction_from_hit(hit, "reaction_similarity")
            for hit in query_qdrant_collection(
                qdrant,
                resolved.transform_collection,
                transform_vector,
                query_limit,
            )
        ]
    except Exception as exc:
        error_text = str(exc)
        if resolved.transform_collection in error_text and "doesn't exist" in error_text:
            warnings.append(
                "Hybrid transform retrieval is not indexed yet, so the app is using molecule retrieval only. "
                "Run `python scripts/index_uspto50k_to_qdrant.py --recreate` to create `reaction_transforms`."
            )
        else:
            warnings.append(
                "Reaction transform retrieval unavailable; using molecule retrieval only. "
                f"Details: {exc}"
            )
        transform_hits = []

    target_classes = infer_target_reaction_classes(smiles)
    reactions = merge_retrieval_hits(
        molecule_hits,
        transform_hits,
        target_classes,
        resolved.weights,
    )
    return RetrievalResult(reactions=reactions[:top_k], warnings=warnings)


def retrieve_reactions_for_smiles(
    smiles: str,
    top_k: int,
    client: QdrantClient | None = None,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    return hybrid_retrieve_reactions_for_smiles(
        smiles,
        top_k=top_k,
        client=client,
        config=config,
    )
