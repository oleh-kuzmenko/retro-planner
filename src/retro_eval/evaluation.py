"""Retrosynthesis prediction metrics: RDKit validity and exact match.

Pure, provider-agnostic functions shared by every `scripts/models/run_*.py`
runner and by `scripts/aggregate_results.py`; kept dependency-free (no LLM
provider, no Qdrant) so they can be unit tested in isolation.
"""

from __future__ import annotations

from retro_eval.chemistry import canonicalize_smiles


def canonical_precursor_set(smiles_list: list[str]) -> frozenset[str] | None:
    """Canonicalize a list of reactant SMILES into an order-independent set.

    Returns None if any fragment fails to parse, since an unparseable
    prediction can never exact-match a reference.
    """
    canonical: set[str] = set()
    for smiles in smiles_list:
        result = canonicalize_smiles(smiles)
        if result is None:
            return None
        canonical.add(result)
    return frozenset(canonical)


def is_exact_match(predicted: list[str], reference: list[str]) -> bool:
    """Whether the predicted precursors are the same set as the reference precursors."""
    predicted_set = canonical_precursor_set(predicted)
    reference_set = canonical_precursor_set(reference)
    if predicted_set is None or reference_set is None:
        return False
    return predicted_set == reference_set


def structure_success_rate(smiles_list: list[str]) -> float:
    """Fraction of predicted SMILES strings that RDKit can successfully parse.

    Each entry may be a single molecule or a dot-joined multi-fragment
    precursor set; both parse through the same RDKit canonicalization path.
    An empty or falsy entry (e.g. a candidate with no usable answer) counts
    as a parse failure.
    """
    if not smiles_list:
        return 0.0

    valid = sum(
        1 for smiles in smiles_list if smiles and canonicalize_smiles(smiles) is not None
    )
    return valid / len(smiles_list)


def split_fragments(smiles: str | None) -> list[str]:
    """Split a dot-joined multi-fragment SMILES prediction into its parts."""
    if not smiles:
        return []
    return [part for part in smiles.split(".") if part]


def is_valid_smiles(smiles: str | None) -> bool:
    """RDKit-parseable, via the same canonicalization path used for exact-match scoring."""
    return bool(smiles) and canonicalize_smiles(smiles) is not None


def is_exact_match_smiles(predicted: str | None, reference: str | None) -> bool:
    """`is_exact_match` for dot-joined prediction/reference strings instead of lists."""
    if not predicted or not reference:
        return False
    return is_exact_match(split_fragments(predicted), split_fragments(reference))
