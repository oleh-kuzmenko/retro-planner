"""Prompt construction for the single-step CoT retrosynthesis contract (PZ section 3.3).

Every function here must only ever produce English text: the target model, the
USPTO-50K/ORD training data it saw, and reaction nomenclature are all English,
and mixing languages in the prompt body degrades generation quality. UI labels
and docstrings may stay in Ukrainian elsewhere in the app; this module may not.
"""


def _score_value(reaction: dict, key: str) -> float:
    try:
        return float(reaction.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _format_examples(rag_examples: list[dict]) -> str:
    if not rag_examples:
        return "No similar reaction precedents were found."

    blocks = []
    for idx, reaction in enumerate(rag_examples, start=1):
        blocks.append(
            "\n".join([
                f"Example {idx} (hybrid_score={_score_value(reaction, 'final_hybrid_score'):.4f})",
                f"reaction_smiles: {reaction.get('reaction_smiles')}",
                f"product_smiles: {reaction.get('product_smiles')}",
                f"reactants_smiles: {reaction.get('reactants_smiles')}",
                f"reaction_class: {reaction.get('reaction_class') or 'N/A'}",
            ])
        )
    return "\n\n".join(blocks)


def build_cot_prompt(target_smiles: str, rag_examples: list[dict] | None = None) -> str:
    """Build the 4-block [System]/[Context]/[Instruction]/[Input] CoT prompt.

    This is the direct English translation of the template in the PZ (page 32):
    a single retrosynthetic step, reasoning inside <think>, only dot-separated
    reactant SMILES inside <answer>.
    """
    examples_block = _format_examples(rag_examples or [])
    return f"""[System] You are an expert organic chemist. Perform a single-step retrosynthetic analysis for the given target molecule.
[Context] The following similar reaction precedents were retrieved from the database:
{examples_block}
[Instruction] Analyze the target molecule step by step inside <think>...</think> tags: identify functional groups, likely reaction centers, and the thermodynamic/kinetic feasibility of the proposed bond disconnection. After that, output only the reactant SMILES strings separated by a dot inside <answer>...</answer> tags, with no other text.
[Input] Target molecule (SMILES): {target_smiles}"""


def build_cot_repair_prompt(
    target_smiles: str,
    rag_examples: list[dict] | None,
    previous_response: str,
    issues: list[str],
) -> str:
    """Build a repair prompt for the same think/answer contract.

    Used when the previous response's <answer> was missing or chemically
    invalid (RDKit parse failure). Keeps the identical tag contract so the
    same parser and validator handle the repaired response.
    """
    examples_block = _format_examples(rag_examples or [])
    issues_block = "\n".join(f"- {issue}" for issue in issues) or "- The answer was missing or unparseable."
    return f"""[System] You are an expert organic chemist correcting a failed single-step retrosynthetic analysis.
[Context] The following similar reaction precedents were retrieved from the database:
{examples_block}

The previous response for this target was:
{previous_response}

It was rejected for these reasons:
{issues_block}
[Instruction] Analyze the target molecule again step by step inside <think>...</think> tags, fixing the issues above. After that, output only valid, RDKit-parseable reactant SMILES strings separated by a dot inside <answer>...</answer> tags, with no other text.
[Input] Target molecule (SMILES): {target_smiles}"""
