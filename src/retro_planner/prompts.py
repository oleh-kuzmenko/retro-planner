import json


ROUTE_JSON_SCHEMA = """
    {
      "routes": [
        {
          "route_name": "Short route name",
          "strategy": "Main retrosynthetic strategy",
          "summary": {
            "difficulty": "low/medium/high plus short reason",
            "cleanliness": "cleanest/average/messy plus short reason",
            "cost": "cheap/moderate/expensive plus short reason",
            "expected_overall_yield": "estimated percent or range",
            "major_risks": ["risk 1", "risk 2"],
            "best_for": "When this route is preferable"
          },
          "steps": [
            {
              "step_number": 1,
              "reaction_name": "Named or descriptive reaction",
              "reactants": ["SMILES_A", "SMILES_B"],
              "product_smiles": "SMILES_PRODUCT",
              "stoichiometry": "Example: A 1.0 equiv, B 1.2 equiv, base 2.0 equiv",
              "reagents": "Catalysts, bases, acids, oxidants, additives",
              "solvent": "Solvent or solvent mixture",
              "temperature_celsius": "temperature or range",
              "reaction_time": "time or range",
              "atmosphere": "air, nitrogen, argon, oxygen-free, etc.",
              "workup_purification": "Quench, extraction, chromatography, crystallization, etc.",
              "expected_yield_percent": "estimated percent or range",
              "important_conditions": "Other important operational details",
              "rationale": "Why this step works"
            }
          ],
          "objective_fit": "How this route addresses the selected optimization objective",
          "evidence_reaction_ids": ["reaction_id_1", "reaction_id_2"]
        }
      ],
      "overall_recommendation": "Which route is cheapest, cleanest, easiest, and strongest overall."
    }
"""


def build_no_rag_system_prompt() -> str:
    return f"""
    You are an expert Organic Chemist specializing in Retrosynthesis.
    Your task is to propose multiple practical retrosynthetic route options for the given Target Molecule.

    CRITICAL RULES:
    1. Output MUST be valid JSON only.
    2. Use standard SMILES strings for all molecules.
    3. Verify that each step's reactants can plausibly form the stated product.
    4. Do not output Markdown formatting like ```json.
    5. Use literature-like reaction conditions. If exact values are uncertain, provide realistic approximate ranges.
    6. Each route must contain 1 to 5 forward synthesis steps ending at the target molecule.
    7. The final step's product_smiles MUST be the target molecule SMILES after canonicalization.
    8. Each step rationale must explain why the chemistry forms the stated product.

    JSON STRUCTURE:
    {ROUTE_JSON_SCHEMA}
    """


def build_no_rag_user_prompt(target_smiles: str, route_count: int) -> str:
    return (
        f"Target Molecule SMILES: {target_smiles}\n"
        "Every route must end with this exact target molecule as the final product.\n"
        f"Generate exactly {route_count} distinct route options."
    )


def build_rag_system_prompt() -> str:
    return f"""
    You are an expert organic chemist specializing in retrosynthesis.
    Use retrieved reaction examples as context for grounded multi-route retrosynthetic planning.
    Retrieved examples are analogies and evidence, not routes to copy.
    If a retrieved example makes a different product, adapt the reaction logic to the target molecule.
    Every route's final step product_smiles MUST be the target molecule SMILES after canonicalization.
    Every rationale must explain target-specific adaptation when retrieved evidence is used.

    Return valid JSON only, without Markdown fences.
    JSON STRUCTURE:
    {ROUTE_JSON_SCHEMA}
    """


def _score_value(reaction: dict, key: str) -> float:
    try:
        return float(reaction.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_rag_prompt(
    target_smiles: str,
    reactions: list[dict],
    optimization_objective: str,
    route_count: int,
) -> str:
    context_lines = []
    for idx, reaction in enumerate(reactions, start=1):
        context_lines.append(
            "\n".join([
                f"Example {idx}",
                f"reaction_id: {reaction.get('reaction_id')}",
                f"molecule_similarity: {_score_value(reaction, 'molecule_similarity'):.4f}",
                f"reaction_similarity: {_score_value(reaction, 'reaction_similarity'):.4f}",
                f"reaction_class_similarity: {_score_value(reaction, 'reaction_class_similarity'):.4f}",
                f"final_hybrid_score: {_score_value(reaction, 'final_hybrid_score'):.4f}",
                f"reaction_smiles: {reaction.get('reaction_smiles')}",
                f"reactants_smiles: {reaction.get('reactants_smiles')}",
                f"product_smiles: {reaction.get('product_smiles')}",
                f"reaction_class: {reaction.get('reaction_class')}",
                f"solvent: {reaction.get('solvent')}",
                f"temperature_celsius: {reaction.get('temperature_celsius')}",
                f"pressure_atm: {reaction.get('pressure_atm')}",
                f"reaction_time_hours: {reaction.get('reaction_time_hours')}",
                f"yield_percent: {reaction.get('yield_percent')}",
                f"source: {reaction.get('source')}",
            ])
        )

    context = "\n\n".join(context_lines)
    return f"""
Target molecule SMILES:
{target_smiles}

Optimization objective:
{optimization_objective}

Hybrid-retrieved reactions:
{context}

Use the hybrid-retrieved examples as evidence, prioritizing reactions with higher final_hybrid_score. Do not copy them blindly if they are chemically mismatched.
Retrieved product_smiles values are example products only. They are not the requested target unless they exactly match the target molecule SMILES above.
Generate exactly {route_count} distinct retrosynthetic route options for the target molecule.
Each route must describe a forward synthesis from simpler commercially available or commonly available starting materials to the target.
Each route must contain 1 to 5 steps.
The final step in every route must have product_smiles equal to the target molecule SMILES above after canonicalization.
Each step must include reactants, product, reaction name, stoichiometric proportions/equivalents, solvent, temperature, time, catalyst or reagent loading, atmosphere, workup or purification, expected yield, and any other important organic reaction conditions.
Each step rationale must explain how the chemistry applies to the target. If a retrieved reaction is used, explain how it was adapted rather than copied.
Summarize each route with difficulty, cleanliness, cost, expected overall yield, major risks, and best use case.
Prefer conditions and reactants that best satisfy the optimization objective.
"""


def build_repair_system_prompt() -> str:
    return """
    You are an expert organic chemist correcting retrosynthesis JSON.
    Return valid JSON only, without Markdown fences.
    The previous answer included routes whose final product was not the requested target.
    Replace off-target routes with chemically plausible routes to the requested target.
    Retrieved reactions are evidence only. Do not copy a retrieved reaction if it makes another product.
    Every route's final step product_smiles MUST be the target molecule SMILES after canonicalization.
    Each rationale must explain how any retrieved evidence was adapted to the target.
    Preserve the same JSON schema: routes with summaries, steps, objective_fit, evidence_reaction_ids, and overall_recommendation.
    """


def build_repair_prompt(
    target_smiles: str,
    reactions: list[dict],
    original_result: dict,
    off_target_routes: list[dict],
    optimization_objective: str,
    route_count: int,
) -> str:
    return f"""
Target molecule SMILES:
{target_smiles}

Optimization objective:
{optimization_objective}

Off-target routes to repair:
{json.dumps(off_target_routes, indent=2)}

Original generated JSON:
{json.dumps(original_result, indent=2)}

Retrieved reaction context:
{build_rag_prompt(target_smiles, reactions, optimization_objective, route_count)}

Return exactly {route_count} routes. Every final step product_smiles must be the target molecule SMILES above.
"""
