"""Response-parsing helpers for the eval scripts' two output contracts.

Two contracts are in play across `scripts/models/run_*.py`:
- `<think>/<answer>` CoT tags (`parse_cot_answer`), reusing the same
  `reasoning.py` parser/validator the RAG+CoT pipeline (`planning.py`) uses.
- A final `Answer: <dot-separated SMILES>` line (`parse_chemllm_answer`) for
  ChemLLM, which otherwise tends to return explanatory prose.
- A compact JSON object (`parse_json_reactants_response`), matching the
  fine-tuned Qwen LoRA adapter's `{"reactants": [...], "reaction_class": ...}`
  training contract.
"""

from __future__ import annotations

import json
import re

from retro_eval.chemistry import canonicalize_smiles
from retro_eval.reasoning import parse_reasoning_response, validate_precursors


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
