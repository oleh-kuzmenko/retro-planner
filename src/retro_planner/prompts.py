import json


REACTION_JSON_SCHEMA = """
    {
      "reaction_name": "Named or descriptive reaction",
      "reactants": ["SMILES_A", "SMILES_B"],
      "product_smiles": "SMILES_PRODUCT",
      "stoichiometry": "Example: A 1.0 equiv, B 1.2 equiv, base 2.0 equiv",
      "reagents": "Catalysts, bases, acids, oxidants, additives",
      "solvent": "Solvent or solvent mixture",
      "temperature_celsius": "temperature or range with units, e.g. 80 deg C or reflux",
      "reaction_time": "time or range with units, e.g. 6 h or overnight",
      "atmosphere": "air, nitrogen, argon, oxygen-free, etc.",
      "workup_purification": "Quench, extraction, chromatography, crystallization, etc.",
      "expected_yield_percent": "estimated percent or range with percent sign",
      "important_conditions": "Other important operational details",
      "rationale": "Why this single reaction forms the target",
      "objective_fit": "How this reaction addresses the selected optimization objective without contradicting the expected yield",
      "evidence_reaction_ids": ["reaction_id_1", "reaction_id_2"]
    }
"""

ROUTES_JSON_SCHEMA = """
    {
      "routes": [
        {
          "route_name": "Short descriptive route name",
          "summary": "Why this route is distinct from the other options",
          "steps": [
            {
              "reaction_name": "Named or descriptive reaction",
              "reactants": ["SMILES_A", "SMILES_B"],
              "product_smiles": "SMILES_PRODUCT",
              "stoichiometry": "Example: A 1.0 equiv, B 1.2 equiv, base 2.0 equiv",
              "reagents": "Catalysts, bases, acids, oxidants, additives",
              "solvent": "Solvent or solvent mixture",
              "temperature_celsius": "temperature or range with units, e.g. 80 deg C or reflux",
              "reaction_time": "time or range with units, e.g. 6 h or overnight",
              "atmosphere": "air, nitrogen, argon, oxygen-free, etc.",
              "workup_purification": "Quench, extraction, chromatography, crystallization, etc.",
              "expected_yield_percent": "estimated percent or range with percent sign",
              "important_conditions": "Other important operational details",
              "rationale": "Why this reaction forms the step product",
              "objective_fit": "How this route addresses the selected optimization objective without contradicting the expected yield or other routes",
              "evidence_reaction_ids": ["reaction_id_1", "reaction_id_2"]
            }
          ],
          "objective_fit": "How this route addresses the selected optimization objective without contradicting the expected yield or other routes",
          "evidence_reaction_ids": ["reaction_id_1", "reaction_id_2"]
        }
      ],
      "overall_recommendation": "Which returned route is preferred and why, consistent with objective_fit and expected yields"
    }
"""


def route_count_instruction(route_count: int) -> str:
    bounded_count = max(1, min(int(route_count), 5))
    return (
        f"Generate exactly {bounded_count} distinct route option"
        f"{'' if bounded_count == 1 else 's'}."
    )


def build_no_rag_system_prompt(route_count: int = 1) -> str:
    return f"""
    You are an expert Organic Chemist specializing in Retrosynthesis.
    Your task is to propose practical alternative forward synthesis routes that make the given Target Molecule.

    CRITICAL RULES:
    1. Output MUST be valid JSON only.
    2. Use standard SMILES strings for all molecules.
    3. Verify that each route can plausibly form the stated final product.
    4. Do not output Markdown formatting like ```json.
    5. Use literature-like reaction conditions. If exact values are uncertain, provide realistic approximate ranges.
    6. {route_count_instruction(route_count)}
    7. The final step product_smiles in every route MUST be the target molecule SMILES after canonicalization.
    8. Each route must be genuinely distinct: different disconnection, starting material class, or key reaction type.
    9. Avoid ambiguous reagent descriptions: if SMILES CO is methanol, call it methanol or MeOH in conditions; only use carbon monoxide when the intended reagent is truly carbon monoxide.
    10. The named or descriptive reaction must include the catalysts, activators, bases, acids, or other reagents that make the stated chemistry plausible.
    11. Temperature values must include units or clear descriptors such as room temperature, reflux, or ambient.
    12. Do not write generic rationales; explain the target-specific bond formation and functional group compatibility.
    13. Prefer one-step routes when plausible, but use multiple steps if that makes a route more realistic.
    14. If optimizing for yield, do not claim a route has the highest yield unless its expected_yield_percent is equal to or higher than the other returned routes.
    15. The overall_recommendation must name one of the returned routes and must not contradict any route objective_fit text.

    JSON STRUCTURE:
    {ROUTES_JSON_SCHEMA}
    """


def build_no_rag_user_prompt(target_smiles: str, route_count: int = 1) -> str:
    return (
        f"Target Molecule SMILES: {target_smiles}\n"
        "Every route's final step product_smiles must be this exact target molecule.\n"
        f"{route_count_instruction(route_count)}"
    )


def build_rag_system_prompt(route_count: int = 1) -> str:
    return f"""
    You are an expert organic chemist specializing in retrosynthesis.
    Use retrieved reaction examples as context for proposing grounded alternative routes.
    Retrieved examples are analogies and evidence, not reactions to copy blindly.
    If a retrieved example makes a different product, adapt the reaction logic to the target molecule.
    The final step product_smiles in every route MUST be the target molecule SMILES after canonicalization.
    The rationale must explain target-specific adaptation when retrieved evidence is used.
    {route_count_instruction(route_count)}
    Each route must be genuinely distinct: different disconnection, starting material class, or key reaction type.
    Avoid ambiguous reagent descriptions: if SMILES CO is methanol, call it methanol or MeOH in conditions; only use carbon monoxide when the intended reagent is truly carbon monoxide.
    The named or descriptive reaction must include the catalysts, activators, bases, acids, or other reagents that make the chemistry plausible.
    Temperature values must include units or clear descriptors such as room temperature, reflux, or ambient.
    Do not write generic rationales like "adapted from Example 1"; explain the target-specific bond formation and functional group compatibility.
    Prefer one-step routes when plausible, but use multiple steps if that makes a route more realistic.
    If optimizing for yield, do not claim a route has the highest yield unless its expected_yield_percent is equal to or higher than the other returned routes.
    The overall_recommendation must name one of the returned routes and must not contradict any route objective_fit text.

    Return valid JSON only, without Markdown fences.
    JSON STRUCTURE:
    {ROUTES_JSON_SCHEMA}
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
    route_count: int = 1,
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
{route_count_instruction(route_count)}
Generate practical forward route options from simpler commercially available or commonly available reactants to the target.
Every route's final step must have product_smiles equal to the target molecule SMILES above after canonicalization.
Each route must include reactants, product, reaction name, stoichiometric proportions/equivalents, solvent, temperature, time, catalyst or reagent loading, atmosphere, workup or purification, expected yield, and any other important organic reaction conditions.
The rationale must explain how the chemistry applies to the target. If a retrieved reaction is used, explain how it was adapted rather than copied.
If a reactant SMILES is CO, describe it as methanol/MeOH in stoichiometry or conditions; do not let it be confused with carbon monoxide.
For the named or descriptive reaction, include the catalysts, activators, bases, acids, or other reagents that make the reaction plausible.
Temperature values must include units or clear descriptors.
Reaction time values must include units. Expected yield values must include a percent sign or explicit percent wording.
Use objective_fit to explain how the reaction satisfies the optimization objective.
If the objective is highest yield, compare expected_yield_percent values consistently across returned routes.
The overall_recommendation must refer to one of the returned route names and must not contradict route objective_fit text.
"""


def build_repair_system_prompt() -> str:
    return """
    You are an expert organic chemist correcting retrosynthesis JSON.
    Return valid JSON only, without Markdown fences.
    The previous answer's product_smiles was missing or not the requested target.
    Replace it with one chemically plausible reaction to the requested target.
    Retrieved reactions are evidence only. Do not copy a retrieved reaction if it makes a different product.
    product_smiles MUST be the target molecule SMILES after canonicalization.
    The rationale must explain how any retrieved evidence was adapted to the target.
    Fix ambiguous methanol/carbon monoxide wording, unsupported reaction conditions, missing temperature units, and generic "adapted from Example N" rationales.
    Preserve the routes JSON schema with objective_fit and evidence_reaction_ids.
    """


def build_repair_prompt(
    target_smiles: str,
    reactions: list[dict],
    original_result: dict,
    optimization_objective: str,
    route_count: int = 1,
) -> str:
    return f"""
Target molecule SMILES:
{target_smiles}

Optimization objective:
{optimization_objective}

Original generated JSON:
{json.dumps(original_result, indent=2)}

Retrieved reaction context:
{build_rag_prompt(target_smiles, reactions, optimization_objective, route_count)}

{route_count_instruction(route_count)}
Every route's final step product_smiles must be the target molecule SMILES above.
"""
