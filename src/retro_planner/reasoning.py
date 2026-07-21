"""Parsing and chemical validation for the <think>/<answer> LLM response contract.

Per the PZ (section 3.3-3.4), a single retrosynthesis generation returns free text
with the full analysis inside <think> (or <reason>) tags and only dot-separated
reactant SMILES inside <answer> tags. Legacy seq2seq-style providers that cannot
produce tags (e.g. ReactionT5) are treated as a valid "no-think" mode: the entire
response is read as the answer.
"""

import logging
import re
from dataclasses import dataclass

from rdkit import Chem, rdBase

from retro_planner.chemistry import canonicalize_smiles

LOGGER = logging.getLogger(__name__)

THINK_TAGS = ("think", "reason")
ANSWER_TAG = "answer"

# Rough allowance for small byproducts (e.g. H2O, HCl, N2) lost when precursors
# combine into the product; below this margin the disconnection likely does not
# conserve mass.
LEAVING_GROUP_HEAVY_ATOM_TOLERANCE = 3


@dataclass(frozen=True)
class ReasoningResult:
    think: str | None
    answer_smiles: list[str]
    raw: str


def _extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).strip()


def _split_answer_smiles(answer_text: str) -> list[str]:
    cleaned = answer_text.strip()
    if ">>" in cleaned:
        cleaned = cleaned.split(">>", 1)[0].strip()
    cleaned = "".join(cleaned.split())
    return [part for part in cleaned.split(".") if part]


def parse_reasoning_response(text: str) -> ReasoningResult:
    """Parse a raw LLM response into its reasoning trace and answer SMILES.

    Falls back to treating the entire response as the answer when no <answer>
    tag is present, so legacy no-think providers stay a supported code path
    rather than an error.
    """
    raw = text or ""

    think = None
    for tag in THINK_TAGS:
        think = _extract_tag(raw, tag)
        if think is not None:
            break

    answer_text = _extract_tag(raw, ANSWER_TAG)
    if answer_text is None:
        LOGGER.warning(
            "No <answer> tag found in LLM response; treating the entire "
            "response as the answer (legacy no-think mode)."
        )
        answer_text = raw.strip()

    return ReasoningResult(
        think=think,
        answer_smiles=_split_answer_smiles(answer_text),
        raw=raw,
    )


def _heavy_atom_count(smiles: str) -> int:
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def validate_precursors(
    answer_smiles: list[str],
    target_smiles: str,
) -> tuple[list[str] | None, list[str], list[str]]:
    """Chemically validate the reactant SMILES parsed out of <answer>.

    Checks RDKit parseability/valence of every fragment and a rough mass-balance
    comparison against the target. Returns (canonical_precursors_or_None,
    warnings, errors); precursors is None whenever the answer cannot be trusted.
    """
    warnings: list[str] = []
    errors: list[str] = []

    if not answer_smiles:
        return None, warnings, [
            "The LLM response did not include any reactant SMILES in <answer> tags."
        ]

    canonical_precursors: list[str] = []
    for fragment in answer_smiles:
        canonical = canonicalize_smiles(fragment)
        if canonical is None:
            errors.append(
                f"Reactant SMILES '{fragment}' failed RDKit validation "
                "(invalid valence or syntax)."
            )
        else:
            canonical_precursors.append(canonical)

    if errors:
        return None, warnings, errors

    canonical_target = canonicalize_smiles(target_smiles)
    if canonical_target:
        precursor_heavy_atoms = sum(
            _heavy_atom_count(smiles) for smiles in canonical_precursors
        )
        target_heavy_atoms = _heavy_atom_count(canonical_target)
        if precursor_heavy_atoms + LEAVING_GROUP_HEAVY_ATOM_TOLERANCE < target_heavy_atoms:
            warnings.append(
                "Precursor heavy-atom count "
                f"({precursor_heavy_atoms}) is well below the target's "
                f"({target_heavy_atoms}); this disconnection may not conserve mass."
            )

    return canonical_precursors, warnings, errors
