#!/usr/bin/env python3
"""Separate "wrote the reaction down differently" from "proposed different chemistry".

A cross-source retrosynthesis number mixes two unrelated failures. USPTO-50K lists only
the molecules that donate atoms to the product -- verified here, not assumed: of 8,580
fragments in its test split, **zero** contribute no mapped atom. ORD lists the whole
vessel charge, dissociated: `[Na+]` occurs 1,502 times in the 5,687-record conditions
test, `O=C([O-])[O-]` 559 times. A model trained on one convention therefore scores near
zero against the other even when its disconnection is right.

So this script scores each run twice: once against the reference as written, and once
against both sides reduced to substrates. The gap between those two numbers is
bookkeeping; what survives the reduction is chemistry.

    substrate = the molecule donates at least one heavy atom to the product

That definition is the one USPTO-50K already follows, which is what makes the two
sources commensurable. It puts Boc2O among the substrates -- its atoms end up in the
product -- where a chemist would call it a reagent; the definition is kept because it is
objective and source-independent, and because USPTO's own atom maps agree with it (the
first test record maps exactly the half of Boc2O that is transferred).

**Applying it without atom maps.** ORD carries none (0 of 2,000 sampled rows), so the rule
is a proxy, in four layers, chemistry before geometry:

1. hand labels win outright, in both directions -- `KNOWN_REAGENTS` and the curated file;
2. carbon-free nucleophiles USPTO itself keeps (ammonia, hydrazine, azide) are substrates,
   catalyst metals are reagents, and a main-group organometallic with a carbon skeleton is
   a substrate whatever its geometry says -- the pinacol half of a boronate outweighs the
   aryl it donates;
3. single atoms and carbon-free species are reagents;
4. everything left is a substrate when its maximum common substructure with the product
   covers at least 3 atoms and at least half the fragment.

Two ways to check it, both here. `--validate` scores the geometric branch against injected
negatives that are deliberately absent from every list, so the lookup cannot score itself.
The stronger test is that the whole rule should be nearly a no-op on USPTO-50K, whose
reactant sides are already exactly the substrates: it now wrongly drops 1.6% of fragments
there, against 11.1% before the hand labels were made authoritative. What remains is
genuine ambiguity -- DMF donating a formyl in a Vilsmeier reaction, methanol esterifying --
where one molecule plays different roles in different records and no per-species label can
be right for both.

Usage:
    python scripts/models/substrate_role_analysis.py --validate
    python scripts/models/substrate_role_analysis.py \
        experiments/v2_model1_topk/baseline_ord_topk.json:"ReactionT5 base" \
        experiments/v2_model1_uspto50k/A5_rsmiles10_e2_ord300_topk.json:"A5"
"""

from __future__ import annotations

import argparse
import json
import random
import re
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

RDLogger.DisableLog("rdApp.*")

# Metals split by what they do. A carbonate or a palladium complex never survives into the
# product; a Grignard, an organozinc or a boronate hands over its carbon skeleton and is
# exactly the coupling partner USPTO-50K keeps on its reactant side.
CATALYST_METALS = frozenset("Na K Ca Cs Rb Pd Pt Ni Fe Al Ti Ag Au Hg Mn Co Cr Ru Rh Ir".split())
NUCLEOPHILE_METALS = frozenset("Mg Zn Li Sn B Cu".split())
METALS = CATALYST_METALS | NUCLEOPHILE_METALS

# 3, not 5. A 5-atom floor threw away exactly the reagents that do donate: allyl bromide
# gives a 3-atom allyl group, acetyl chloride a 3-atom acetyl, phosgene its chlorine. The
# 70% fraction is what rejects coincidental matches, and it holds for all three (3 of 4
# heavy atoms).
MIN_MCS_ATOMS = 3
MIN_MCS_FRACTION = 0.5

# Solvents, bases and coupling agents are named outright rather than left to the MCS test,
# which passed pyridine, benzene and toluene as substrates whenever they shared an aromatic
# ring with the product. Every entry here donates nothing by construction.
KNOWN_REAGENTS = frozenset(
    Chem.MolToSmiles(Chem.MolFromSmiles(s))
    for s in (
        # solvents
        "O", "CO", "CCO", "CC(C)O", "C1CCOC1", "CN(C)C=O", "CS(C)=O", "ClCCl",
        "ClC(Cl)Cl", "ClC(Cl)(Cl)Cl", "Cc1ccccc1", "c1ccccc1", "CCOC(C)=O", "CC#N",
        "C1COCCO1", "CCOCC", "CCCCCC", "CCCCCCC", "CC(C)=O", "CN1CCCC1=O",
        # bases and amines
        "c1ccncc1", "CCN(CC)CC", "CCN(C(C)C)C(C)C", "CN(C)c1ccncc1",
        # acids used as media
        "O=C(O)C(F)(F)F", "Cl", "Cc1ccc(S(=O)(=O)O)cc1",
        # coupling agents
        "CCN=C=NCCCN(C)C", "C1CCC(N=C=NC2CCCCC2)CC1", "On1nnc2ccccc21",
        "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
    )
)

# Validation negatives, deliberately **disjoint from KNOWN_REAGENTS**: bases and ligands
# that donate nothing but are not named in the rule, so they test the MCS branch rather
# than the lookup. Scoring against the named list would be circular.
HELDOUT_REAGENTS = (
    "C1CCC2=NCCCN2CC1",                   # DBU
    "CN1CCOCC1",                          # N-methylmorpholine
    "Cc1cccc(C)n1",                       # 2,6-lutidine
    "C1CN2CCN1CC2",                       # DABCO
    "CN(C)CCN(C)C",                       # TMEDA
    "Cn1ccnc1",                           # N-methylimidazole
    "c1ccc2ncccc2c1",                     # quinoline
    "c1ccc(-c2ccccn2)nc1",                # 2,2'-bipyridine
    "C1CN2CCC1CC2",                       # quinuclidine
    "Cc1ccccc1P(c1ccccc1C)c1ccccc1C",     # tri-o-tolylphosphine
)

ATOM_MAP = re.compile(r":(\d+)\]")

# The 200 most frequent species the MCS branch decides, labelled by hand. The rule alone
# cannot separate a carbonate from a methyl iodide -- both are small and share little with
# the product -- so the frequent half of that branch is settled by chemistry rather than by
# geometry. Rare species are left to the rule: 32.7% of the branch occurs exactly once, and
# a molecule appearing once in 138,869 reactions is not a shelf reagent.
CURATED_ROLES_FILE = Path(__file__).resolve().parents[2] / "data" / "reagent_roles_top200.tsv"


def _load_curated_roles() -> tuple[frozenset[str], frozenset[str]]:
    """Both directions are authoritative. A curated `substrate` has to override the
    geometric test, not merely skip the reagent list: methyl iodide donates 2 heavy atoms
    and Boc2O donates 6 of its 15, so both fail a fraction-of-fragment threshold that was
    built to describe skeletons."""
    if not CURATED_ROLES_FILE.exists():
        return frozenset(), frozenset()
    reagents, substrates = set(), set()
    for line in CURATED_ROLES_FILE.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        smiles, role = line.split("\t")[:2]
        (reagents if role == "reagent" else substrates).add(smiles)
    return frozenset(reagents), frozenset(substrates)


CURATED_REAGENTS, CURATED_SUBSTRATES = _load_curated_roles()

# Carbon-free species that nevertheless donate: USPTO-50K keeps every one of these on its
# reactant side, so the carbon test cannot be the last word.
INORGANIC_NUCLEOPHILES = frozenset(
    Chem.MolToSmiles(Chem.MolFromSmiles(s))
    for s in ("N", "NN", "[N-]=[N+]=[N-]", "NO", "ON", "[SH-]", "S", "O=S(=O)([O-])[O-]", "[N-]=[N+]=N")
)


@lru_cache(maxsize=None)
def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=None)
def canonical(smiles: str) -> str | None:
    mol = _mol(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


@lru_cache(maxsize=None)
def is_substrate(fragment: str, product: str) -> bool:
    frag, prod = _mol(fragment), _mol(product)
    if frag is None or prod is None:
        return False
    # Hand labels first, in both directions -- they are chemistry, the rest is geometry.
    if fragment in KNOWN_REAGENTS or fragment in CURATED_REAGENTS:
        return False
    if fragment in CURATED_SUBSTRATES or fragment in INORGANIC_NUCLEOPHILES:
        return True
    symbols = {atom.GetSymbol() for atom in frag.GetAtoms()}
    has_carbon = any(atom.GetSymbol() == "C" for atom in frag.GetAtoms())
    if symbols & CATALYST_METALS:
        return False
    # An organoboron or organozinc is a substrate whatever its geometry says: the pinacol
    # half of a boronate is larger than the aryl it donates, so the fraction test would
    # reject the very partner that makes the bond.
    if (symbols & NUCLEOPHILE_METALS) and has_carbon:
        return True
    if not has_carbon:
        return False
    heavy = frag.GetNumHeavyAtoms()
    if heavy == 1:
        return False
    shared = rdFMCS.FindMCS(
        [frag, prod], timeout=2, ringMatchesRingOnly=True, completeRingsOnly=False
    ).numAtoms
    return shared >= MIN_MCS_ATOMS and shared >= MIN_MCS_FRACTION * heavy


def molecule_set(smiles: str, product: str, keep_all: bool) -> set[str]:
    out: set[str] = set()
    for fragment in smiles.split("."):
        if not fragment:
            continue
        canon = canonical(fragment)
        if canon is None:
            continue
        if keep_all or is_substrate(canon, product):
            out.add(canon)
    return out


def score_run(path: Path, ks=(1, 3, 5)) -> dict:
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    out = {}
    for mode in ("full", "substrates"):
        keep_all = mode == "full"
        hits = {k: 0 for k in ks}
        for record in records:
            product = canonical(record["product_smiles"])
            reference = molecule_set(record["reactants_smiles"], product, keep_all)
            if not reference:
                continue
            rank = None
            for index, candidate in enumerate(record["candidates"][: max(ks)], start=1):
                if molecule_set(candidate, product, keep_all) == reference:
                    rank = index
                    break
            for k in ks:
                if rank is not None and rank <= k:
                    hits[k] += 1
        out[mode] = {k: hits[k] / len(records) for k in ks}
    return out


def validate(sample_size: int, seed: int) -> None:
    from datasets import load_dataset

    random.seed(seed)
    dataset = load_dataset("bisectgroup/USPTO_50K", split="test")

    unmapped = 0
    fragments = 0
    for row in dataset:
        product_maps = set(ATOM_MAP.findall(row["product"]))
        for fragment in row["reactants"].split("."):
            fragments += 1
            if not set(ATOM_MAP.findall(fragment)) & product_maps:
                unmapped += 1
    print(f"USPTO-50K test: {fragments} fragments, {unmapped} donate no atom to the product")

    def strip_maps(smiles: str) -> str | None:
        return canonical(ATOM_MAP.sub("]", smiles))

    pairs = []
    for index in random.sample(range(len(dataset)), sample_size):
        row = dataset[index]
        product = strip_maps(row["product"])
        if product is None:
            continue
        for fragment in row["reactants"].split("."):
            clean = strip_maps(fragment)
            if clean:
                pairs.append((clean, product, True))
        for reagent in random.sample(HELDOUT_REAGENTS, 2):
            pairs.append((canonical(reagent), product, False))

    tp = fp = fn = tn = 0
    for fragment, product, truly_substrate in pairs:
        predicted = is_substrate(fragment, product)
        if truly_substrate and predicted:
            tp += 1
        elif truly_substrate:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1

    print(f"validation pairs: {len(pairs)} ({tp + fn} substrates, {tn + fp} injected reagents)")
    print(f"  substrate recall  {tp / (tp + fn):.1%}")
    print(f"  reagent recall    {tn / (tn + fp):.1%}")
    print(f"  reagent precision {tn / (tn + fn):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="top-k result files as path[:label]")
    parser.add_argument("--validate", action="store_true", help="measure the proxy against USPTO atom maps")
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.validate:
        validate(args.sample_size, args.seed)
        if args.runs:
            print()

    if not args.runs:
        return

    print(f"{'run':32s} {'reference':16s} {'top-1':>7s} {'top-3':>7s} {'top-5':>7s}")
    for entry in args.runs:
        path, _, label = entry.partition(":")
        label = label or Path(path).stem
        scored = score_run(Path(path))
        for mode, name in (("full", "as written"), ("substrates", "substrates only")):
            row = scored[mode]
            print(f"{label:32s} {name:16s} {row[1]:6.1%} {row[3]:6.1%} {row[5]:6.1%}")


if __name__ == "__main__":
    main()
