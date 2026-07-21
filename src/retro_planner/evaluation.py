"""Retrosynthesis evaluation metrics (PZ section 4.1-4.2).

Pure, provider-agnostic functions for comparing Zero-shot against RAG+CoT
generation on USPTO-50K: Top-k exact match and Structure Success Rate (the
fraction of predicted SMILES that RDKit can parse). `scripts/evaluate_
retrosynthesis.py` wires these to a live LLM provider and Qdrant retrieval
to produce the section 4.2 table; keeping the metrics here lets them be unit
tested without any network or model dependency.
"""

from __future__ import annotations

from retro_planner.chemistry import canonicalize_smiles


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


def top_k_exact_match(
    predictions: list[list[list[str]]],
    references: list[list[str]],
    k: int,
) -> float:
    """Fraction of targets whose first `k` ranked candidates contain an exact match.

    `predictions[i]` is the ranked list of candidate precursor lists generated
    for target `i` (rank 1 first); `references[i]` is that target's ground-truth
    precursor list. Ranking, not just membership, matters: only the first `k`
    candidates per target are considered.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return 0.0

    hits = sum(
        any(is_exact_match(candidate, reference) for candidate in candidates[:k])
        for candidates, reference in zip(predictions, references)
    )
    return hits / len(predictions)


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


def format_results_table(results: dict[str, dict[str, float]], k_values: list[int]) -> str:
    """Render a Table-4.1-style markdown report: one row per configuration.

    `results` maps a display label (e.g. "Zero-shot", "RAG+CoT") to a metrics
    dict with `num_targets`, `top_{k}` for each `k` in `k_values`, and
    `structure_success_rate`.
    """
    headers = ["Mode", "N"] + [f"Top-{k}" for k in k_values] + ["Structure Success Rate"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]

    for label, metrics in results.items():
        row = [label, str(int(metrics["num_targets"]))]
        row.extend(f"{metrics[f'top_{k}'] * 100:.1f}%" for k in k_values)
        row.append(f"{metrics['structure_success_rate'] * 100:.1f}%")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
