"""Load the USPTO/ORD-format JSON dataset shared by every model-runner script.

Both formats carry `product_smiles`/`reactants_smiles`; ORD records
additionally carry `reaction_id`/`solvent`/`temperature_celsius`/`catalyst`/
`yield_percent`, used by `run_rag_cot_groq.py` to also ask for reaction
conditions (never scored programmatically, only logged for qualitative
review).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

ORD_FIELD_NAMES = ("reaction_id", "solvent", "temperature_celsius", "catalyst", "yield_percent")


@dataclass(frozen=True)
class EvalRecord:
    index: int
    product_smiles: str
    reactants_smiles: str | None
    reaction_id: str | None = None
    solvent: str | None = None
    temperature_celsius: float | None = None
    catalyst: str | None = None
    yield_percent: float | None = None

    @property
    def is_ord(self) -> bool:
        """Whether this record carries ORD condition metadata."""
        return any(getattr(self, name) is not None for name in ORD_FIELD_NAMES)


def load_records(path: Path) -> list[EvalRecord]:
    """Load a USPTO- or ORD-format JSON array, keeping index = original row position.

    Every model-runner script reads the same file the same way, so indices
    line up across each script's `results.json` even if a record is skipped
    (missing `product_smiles`) -- the skip is deterministic given the same input.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"{path} must contain a JSON array of reaction records.")

    records: list[EvalRecord] = []
    for idx, row in enumerate(raw):
        product_smiles = row.get("product_smiles")
        if not product_smiles:
            LOGGER.warning("Skipping record %d in %s: missing product_smiles.", idx, path)
            continue
        records.append(
            EvalRecord(
                index=idx,
                product_smiles=product_smiles,
                reactants_smiles=row.get("reactants_smiles"),
                reaction_id=row.get("reaction_id"),
                solvent=row.get("solvent"),
                temperature_celsius=row.get("temperature_celsius"),
                catalyst=row.get("catalyst"),
                yield_percent=row.get("yield_percent"),
            )
        )
    return records
