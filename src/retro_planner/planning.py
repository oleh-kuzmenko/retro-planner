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
    route_count: int
    optimization_objective: str = "BALANCED"
    reactions: list[dict] | None = None


@dataclass(frozen=True)
class PlanResult:
    result: dict | str | None
    warnings: list[str]
    errors: list[str]


def _route_steps(route: dict) -> list[dict]:
    steps = route.get("steps", [])
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)][:5]


def route_final_product(route: dict) -> str | None:
    steps = _route_steps(route)
    if not steps:
        return None

    raw_product_smiles = steps[-1].get("product_smiles")
    if not raw_product_smiles:
        return None
    return canonicalize_smiles(str(raw_product_smiles))


def route_final_product_matches_target(route: dict, canonical_target: str) -> bool:
    return route_final_product(route) == canonical_target


def off_target_route_summaries(
    result: dict,
    canonical_target: str,
) -> list[dict]:
    routes = result.get("routes")
    if not isinstance(routes, list):
        return []

    summaries = []
    for idx, route in enumerate(routes, start=1):
        if not isinstance(route, dict):
            continue
        final_product = route_final_product(route)
        if final_product != canonical_target:
            summaries.append({
                "route_index": idx,
                "route_name": route.get("route_name"),
                "strategy": route.get("strategy"),
                "final_product_smiles": final_product,
                "target_smiles": canonical_target,
            })
    return summaries


def filter_target_matching_routes(
    result: dict,
    canonical_target: str,
) -> tuple[dict | None, int]:
    routes = result.get("routes")
    if not isinstance(routes, list):
        return result, 0

    route_dicts = [route for route in routes if isinstance(route, dict)]
    valid_routes = [
        route
        for route in route_dicts
        if route_final_product_matches_target(route, canonical_target)
    ]
    removed_count = len(route_dicts) - len(valid_routes)

    if not route_dicts or removed_count == 0:
        return result, 0
    if not valid_routes:
        return None, removed_count

    filtered_result = dict(result)
    filtered_result["routes"] = valid_routes
    return filtered_result, removed_count


def _parse_json_or_text(content: str) -> dict | str:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def get_retrosynthesis_plan(request: GenerationRequest) -> PlanResult:
    try:
        content = request.llm_provider.generate(
            messages=[
                {"role": "system", "content": build_no_rag_system_prompt()},
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

        filtered_result, removed_count = filter_target_matching_routes(
            parsed,
            request.target_smiles,
        )
        warnings = []
        errors = []
        if removed_count:
            warnings.append(
                "Some generated routes were removed because their final product did not match the target."
            )
        if filtered_result is None:
            errors.append(
                "The LLM did not produce any routes whose final product matches the target molecule."
            )
        return PlanResult(result=filtered_result, warnings=warnings, errors=errors)
    except Exception as exc:
        return PlanResult(result=None, warnings=[], errors=[f"API Error: {exc}"])


def repair_off_target_routes_with_llm(
    request: GenerationRequest,
    original_result: dict,
) -> dict | str | None:
    reactions = request.reactions or []
    off_target_routes = off_target_route_summaries(
        original_result,
        request.target_smiles,
    )
    if not off_target_routes:
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
                    off_target_routes=off_target_routes,
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
                {"role": "system", "content": build_rag_system_prompt()},
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

        filtered_result, removed_count = filter_target_matching_routes(
            parsed,
            request.target_smiles,
        )
        if removed_count:
            try:
                repair_result = repair_off_target_routes_with_llm(request, parsed)
            except Exception as exc:
                repair_result = parsed
                repair_warning = f"Could not repair off-target routes: {exc}"
            else:
                repair_warning = None

            if isinstance(repair_result, dict):
                filtered_result, removed_count = filter_target_matching_routes(
                    repair_result,
                    request.target_smiles,
                )
            elif isinstance(repair_result, str):
                return PlanResult(result=repair_result, warnings=[], errors=[])
        else:
            repair_warning = None

        warnings = []
        errors = []
        if repair_warning:
            warnings.append(repair_warning)
        if removed_count:
            warnings.append(
                "Some generated routes were removed because their final product did not match the target."
            )
        if filtered_result is None:
            errors.append(
                "The LLM did not produce any routes whose final product matches the target molecule."
            )
        return PlanResult(result=filtered_result, warnings=warnings, errors=errors)
    except Exception as exc:
        return PlanResult(result=None, warnings=[], errors=[f"LLM API failure: {exc}"])
