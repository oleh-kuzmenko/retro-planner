"""Response-parsing helpers for the eval scripts' two output contracts.

Two contracts are in play across `scripts/models/run_*.py`:
- `<think>/<answer>` CoT tags (`parse_cot_answer`), reusing the same
  `reasoning.py` parser/validator the RAG+CoT pipeline (`planning.py`) uses.
- A final `Answer: <dot-separated SMILES>` line (`parse_chemllm_answer`) for
  ChemLLM, which otherwise tends to return explanatory prose.
- A compact JSON object (`parse_json_reactants_response`), matching the
  fine-tuned Qwen LoRA adapter's `{"reactants": [...], "reaction_class": ...}`
  training contract.
- A `<think>/<answer>` reranking response (`parse_rerank_answer`) for the hybrid
  T5+RAG+LLM pipeline, where `<answer>` must echo one of a fixed set of
  ReactionT5v2 candidates rather than a freshly proposed disconnection.
"""

from __future__ import annotations

import json
import re

from retro_eval.chemistry import canonicalize_smiles
from retro_eval.reasoning import parse_reasoning_response, validate_precursors

_ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict:
    """Pull a JSON object out of a response that may be wrapped in ```json fences."""
    cleaned = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError(f"No JSON object found in response: {cleaned[:300]}")
    return json.loads(cleaned[first : last + 1])


def parse_cot_answer(raw: str, product_smiles: str) -> tuple[str, dict]:
    """Parse a `<think>/<answer>` response into (dot-joined predicted reactants, log extras)."""
    reasoning = parse_reasoning_response(raw)
    precursors, warnings, errors = validate_precursors(reasoning.answer_smiles, product_smiles)
    predicted = ".".join(precursors) if precursors else ".".join(reasoning.answer_smiles)
    return predicted, {"think": reasoning.think, "warnings": warnings, "errors": errors}


def parse_chemllm_answer(raw: str, product_smiles: str) -> tuple[str, dict]:
    """Parse ChemLLM's final ``Answer:`` line without treating prose as SMILES."""
    match = re.search(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$", raw or "")
    if match is None:
        return "", {
            "think": None,
            "candidate_answer": "",
            "warnings": [],
            "errors": ["ChemLLM response did not contain a final 'Answer: <SMILES>' line."],
        }

    answer = match.group(1).strip().strip("`")
    answer_smiles = [fragment for fragment in answer.split(".") if fragment]
    precursors, warnings, errors = validate_precursors(answer_smiles, product_smiles)
    return ".".join(precursors or []), {
        "think": None,
        "candidate_answer": answer,
        "warnings": warnings,
        "errors": errors,
    }


def parse_json_reactants_response(raw: str) -> tuple[str, dict]:
    """Parse a `{"reactants": [...], "reaction_class": ...}` response into (dot-joined predicted reactants, log extras)."""
    data = extract_json_object(raw)
    raw_reactants = data.get("reactants") or []
    canonical = [canonicalize_smiles(fragment) for fragment in raw_reactants]
    valid = [fragment for fragment in canonical if fragment]
    dropped = len(raw_reactants) - len(valid)
    extra = {
        "reaction_class": data.get("reaction_class"),
        "warnings": [f"Dropped {dropped} unparseable reactant fragment(s)."] if dropped else [],
    }
    return ".".join(valid), extra


def _first_valid_candidate(candidates: list[str]) -> str | None:
    return next((candidate for candidate in candidates if canonicalize_smiles(candidate) is not None), None)


def _extract_think(raw: str) -> str | None:
    for tag in ("think", "reason"):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL | re.IGNORECASE)
        if match is not None:
            return match.group(1).strip()
    return None


def parse_rerank_answer(raw: str, candidates: list[str]) -> tuple[str, dict]:
    """Parse the hybrid reranker's `<answer>` tag into one of the T5 candidates it was shown.

    Falls back to the first RDKit-*valid* T5 candidate (not blindly `candidates[0]`)
    whenever the `<answer>` tag is missing or its content doesn't canonicalize to a valid
    molecule -- a beam-search hypothesis ranked below #1 is still a better fallback than an
    invalid one ranked #1. If none of the shown candidates are valid either, returns ""
    with an `errors` entry so the record is honestly flagged as a failed prediction rather
    than silently repeating a broken SMILES string. When the extracted answer is valid but
    not an exact echo of a shown candidate (paraphrased atom order, stray whitespace, ...),
    it's matched back to the candidate it canonically equals; if it matches none, the
    validated literal text is still used and flagged.
    """
    raw = raw or ""
    think = _extract_think(raw)
    match = _ANSWER_TAG_PATTERN.search(raw)

    def _invalid_answer_fallback(reason: str) -> tuple[str, dict]:
        fallback = _first_valid_candidate(candidates)
        if fallback is None:
            return "", {
                "think": think,
                "warnings": [f"{reason} None of the {len(candidates)} T5 candidates are valid SMILES either."],
                "errors": ["No valid candidate available to fall back to."],
            }
        return fallback, {
            "think": think,
            "warnings": [f"{reason} Fell back to the first valid T5 candidate ('{fallback}')."],
        }

    if match is None:
        return _invalid_answer_fallback("LLM response did not contain an <answer> tag.")

    extracted = match.group(1).strip().strip("`")
    canonical_extracted = canonicalize_smiles(extracted)
    if canonical_extracted is None:
        return _invalid_answer_fallback(f"LLM <answer> content '{extracted}' failed RDKit validation.")

    canonical_candidates = {canonicalize_smiles(candidate): candidate for candidate in candidates}
    if canonical_extracted in canonical_candidates:
        return canonical_candidates[canonical_extracted], {"think": think, "warnings": []}

    return extracted, {
        "think": think,
        "warnings": ["LLM <answer> did not exactly match any T5 candidate; using its literal (validated) output."],
    }
