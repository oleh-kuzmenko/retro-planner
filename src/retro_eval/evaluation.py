"""Retrosynthesis prediction metrics: RDKit validity and exact match.

Pure, provider-agnostic functions shared by every `scripts/models/run_*.py`
runner and by `scripts/aggregate_results.py`; kept dependency-free (no LLM
provider, no Qdrant) so they can be unit tested in isolation.
"""

from __future__ import annotations

from retro_eval.chemistry import canonicalize_smiles, is_inorganic_salt_or_catalyst


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


def strip_salts_and_catalysts(fragments: list[str]) -> list[str]:
    """Drop inorganic salt/counter-ion/catalyst fragments (see `is_inorganic_salt_or_catalyst`).

    Fragments that fail to parse are kept rather than dropped, so an invalid
    prediction still fails exact-match downstream instead of being masked by
    having its garbage silently removed.
    """
    return [fragment for fragment in fragments if not is_inorganic_salt_or_catalyst(fragment)]


def is_core_exact_match(predicted: list[str], reference: list[str]) -> bool:
    """`is_exact_match`, ignoring inorganic salts/catalysts on both sides.

    Reaction records (e.g. ORD) often list every reagent used -- bases, drying
    salts, metal catalysts -- alongside the true bond-forming reactants. A model
    that recovers the correct disconnection but omits those auxiliary reagents
    (or a reranker that picks a beam missing them) should still count as correct
    here, since it got the actual retrosynthetic transform right.
    """
    return is_exact_match(strip_salts_and_catalysts(predicted), strip_salts_and_catalysts(reference))


def is_core_exact_match_smiles(predicted: str | None, reference: str | None) -> bool:
    """`is_core_exact_match` for dot-joined prediction/reference strings instead of lists."""
    if not predicted or not reference:
        return False
    return is_core_exact_match(split_fragments(predicted), split_fragments(reference))
