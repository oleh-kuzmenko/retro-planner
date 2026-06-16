import json
from dataclasses import dataclass

from retro_planner.chemistry import canonicalize_smiles
from retro_planner.llm_providers import LLMProvider
from retro_planner.prompts import (
    build_no_rag_system_prompt,
    build_no_rag_user_prompt,
    build_rag_prompt,
    build_rag_system_prompt,
    build_repair_prompt,
    build_repair_system_prompt,
)


@dataclass(frozen=True)
class GenerationRequest:
    target_smiles: str
    llm_provider: LLMProvider
    model: str
    temperature: float
    optimization_objective: str = "BALANCED"
    route_count: int = 1
    reactions: list[dict] | None = None


@dataclass(frozen=True)
class PlanResult:
    result: dict | str | None
    warnings: list[str]
    errors: list[str]


def reaction_product(result: dict) -> str | None:
    raw_product_smiles = result.get("product_smiles")
    if not raw_product_smiles:
        return None
    return canonicalize_smiles(str(raw_product_smiles))


def reaction_product_matches_target(result: dict, canonical_target: str) -> bool:
    return reaction_product(result) == canonical_target


def validate_target_matching_reaction(
    result: dict,
    canonical_target: str,
) -> tuple[dict | None, str | None]:
    product = reaction_product(result)
    if not product:
        return None, "The LLM response did not include a valid product_smiles value."
    if product != canonical_target:
        return None, (
            "The LLM produced a reaction whose product does not match the target molecule."
        )
    return result, None


def _final_step(route: dict) -> dict | None:
    steps = route.get("steps")
    if isinstance(steps, list):
        valid_steps = [step for step in steps if isinstance(step, dict)]
        if valid_steps:
            return valid_steps[-1]
    if route.get("product_smiles"):
        return route
    return None


def validate_target_matching_plan(
    result: dict,
    canonical_target: str,
    route_count: int,
) -> tuple[dict | None, list[str], list[str]]:
    bounded_count = max(1, min(int(route_count), 5))
    routes = result.get("routes")
    if not isinstance(routes, list):
        valid_result, error = validate_target_matching_reaction(result, canonical_target)
        return valid_result, [], [error] if error else []

    valid_routes = []
    warnings = []
    for index, route in enumerate(routes[:bounded_count], start=1):
        if not isinstance(route, dict):
            warnings.append(f"Route {index} was skipped because it was not an object.")
            continue

        final_step = _final_step(route)
        if not final_step:
            warnings.append(f"Route {index} was skipped because it has no usable steps.")
            continue

        _, error = validate_target_matching_reaction(final_step, canonical_target)
        if error:
            warnings.append(f"Route {index} was skipped: {error}")
            continue

        valid_routes.append(route)

    if not valid_routes:
        return None, warnings, [
            "The LLM did not produce any route whose final product matches the target molecule."
        ]

    valid_result = dict(result)
    valid_result["routes"] = valid_routes
    return valid_result, warnings, []


def _parse_json_or_text(content: str) -> dict | str:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def get_retrosynthesis_plan(request: GenerationRequest) -> PlanResult:
    try:
        content = request.llm_provider.generate(
            messages=[
                {
                    "role": "system",
                    "content": build_no_rag_system_prompt(request.route_count),
                },
                {
                    "role": "user",
                    "content": build_no_rag_user_prompt(
                        request.target_smiles,
                        request.route_count,
                    ),
                },
            ],
            model=request.model,
            temperature=request.temperature,
            json_mode=True,
        )
        parsed = _parse_json_or_text(content)
        if not isinstance(parsed, dict):
            return PlanResult(result=parsed, warnings=[], errors=[])

        valid_result, warnings, errors = validate_target_matching_plan(
            parsed,
            request.target_smiles,
            request.route_count,
        )
        return PlanResult(
            result=valid_result,
            warnings=warnings,
            errors=errors,
        )
    except Exception as exc:
        return PlanResult(result=None, warnings=[], errors=[f"API Error: {exc}"])


def repair_off_target_reaction_with_llm(
    request: GenerationRequest,
    original_result: dict,
) -> dict | str | None:
    reactions = request.reactions or []
    if reaction_product_matches_target(original_result, request.target_smiles):
        return original_result

    content = request.llm_provider.generate(
        messages=[
            {"role": "system", "content": build_repair_system_prompt()},
            {
                "role": "user",
                "content": build_repair_prompt(
                    target_smiles=request.target_smiles,
                    reactions=reactions,
                    original_result=original_result,
                    optimization_objective=request.optimization_objective,
                    route_count=request.route_count,
                ),
            },
        ],
        model=request.model,
        temperature=max(0.0, min(request.temperature, 0.2)),
        json_mode=True,
    )
    return _parse_json_or_text(content)


def call_llm_with_rag(request: GenerationRequest) -> PlanResult:
    reactions = request.reactions or []
    try:
        content = request.llm_provider.generate(
            messages=[
                {
                    "role": "system",
                    "content": build_rag_system_prompt(request.route_count),
                },
                {
                    "role": "user",
                    "content": build_rag_prompt(
                        request.target_smiles,
                        reactions,
                        request.optimization_objective,
                        request.route_count,
                    ),
                },
            ],
            model=request.model,
            temperature=request.temperature,
            json_mode=True,
        )
        parsed = _parse_json_or_text(content)
        if not isinstance(parsed, dict):
            return PlanResult(result=parsed, warnings=[], errors=[])

        valid_result, warnings, errors = validate_target_matching_plan(
            parsed,
            request.target_smiles,
            request.route_count,
        )
        repair_warning = None
        if errors:
            try:
                repair_result = repair_off_target_reaction_with_llm(request, parsed)
            except Exception as exc:
                repair_result = parsed
                repair_warning = f"Could not repair off-target reaction: {exc}"
            else:
                repair_warning = None

            if isinstance(repair_result, dict):
                valid_result, repair_warnings, errors = validate_target_matching_plan(
                    repair_result,
                    request.target_smiles,
                    request.route_count,
                )
                warnings.extend(repair_warnings)
            elif isinstance(repair_result, str):
                return PlanResult(result=repair_result, warnings=[], errors=[])

        if repair_warning:
            warnings.append(repair_warning)
        return PlanResult(result=valid_result, warnings=warnings, errors=errors)
    except Exception as exc:
        return PlanResult(result=None, warnings=[], errors=[f"LLM API failure: {exc}"])
