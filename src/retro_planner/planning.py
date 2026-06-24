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

GENERATION_TEMPERATURE = 0.2
REPAIR_TEMPERATURE = 0.0


@dataclass(frozen=True)
class GenerationRequest:
    target_smiles: str
    llm_provider: LLMProvider
    model: str
    optimization_objective: str = "BALANCED"
    route_count: int = 1
    reactions: list[dict] | None = None
    target_smiles_only: bool = False


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


def _single_step(route: dict) -> tuple[dict | None, str | None]:
    steps = route.get("steps")
    if not isinstance(steps, list):
        return None, "it does not include a steps list."

    valid_steps = [step for step in steps if isinstance(step, dict)]
    if len(valid_steps) != 1:
        return (
            None,
            f"it includes {len(valid_steps)} usable steps instead of exactly one.",
        )

    return valid_steps[0], None


def validate_target_matching_plan(
    result: dict,
    canonical_target: str,
    route_count: int,
) -> tuple[dict | None, list[str], list[str]]:
    bounded_count = max(1, min(int(route_count), 5))
    routes = result.get("routes")
    if not isinstance(routes, list):
        return None, [], [
            "The LLM response did not include a routes list with one-step route options."
        ]

    valid_routes = []
    warnings = []
    for index, route in enumerate(routes[:bounded_count], start=1):
        if not isinstance(route, dict):
            warnings.append(f"Route {index} was skipped because it was not an object.")
            continue

        step, step_error = _single_step(route)
        if step_error:
            warnings.append(f"Route {index} was skipped because {step_error}")
            continue

        _, error = validate_target_matching_reaction(step, canonical_target)
        if error:
            warnings.append(f"Route {index} was skipped: {error}")
            continue

        valid_routes.append(route)

    if not valid_routes:
        return None, warnings, [
            "The LLM did not produce any valid one-step reaction to the target molecule."
        ]

    valid_result = dict(result)
    valid_result["routes"] = valid_routes
    return valid_result, warnings, []


def _parse_json_or_text(content: str) -> dict | str:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _reactant_plan_from_text(
    content: str,
    target_smiles: str,
) -> PlanResult:
    raw_output = _strip_code_fence(content)
    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError:
        decoded = raw_output

    if isinstance(decoded, str):
        reactant_text = decoded.strip()
    else:
        return PlanResult(
            result=None,
            warnings=[],
            errors=[
                "Target-only mode expected the model to return reactant SMILES as plain text."
            ],
        )

    if ">>" in reactant_text:
        reactant_text = reactant_text.split(">>", 1)[0].strip()
    reactant_text = "".join(reactant_text.split())
    reactants = [
        canonicalize_smiles(part)
        for part in reactant_text.split(".")
        if part
    ]

    if not reactants or any(reactant is None for reactant in reactants):
        return PlanResult(
            result=None,
            warnings=[],
            errors=[
                "The custom model did not return valid dot-separated reactant SMILES. "
                f"Raw response: {raw_output or '(empty)'}"
            ],
        )

    step = {
        "reaction_name": "Custom model prediction",
        "reactants": reactants,
        "product_smiles": target_smiles,
    }
    return PlanResult(
        result={
            "routes": [
                {
                    "route_name": "Predicted precursors",
                    "summary": "Reactants predicted directly from the target SMILES.",
                    "steps": [step],
                }
            ]
        },
        warnings=[],
        errors=[],
    )


def get_retrosynthesis_plan(request: GenerationRequest) -> PlanResult:
    try:
        if request.target_smiles_only:
            generate_completion = getattr(
                request.llm_provider,
                "generate_completion",
                None,
            )
            if not callable(generate_completion):
                return PlanResult(
                    result=None,
                    warnings=[],
                    errors=[
                        "The selected provider does not support prompt-based completions."
                    ],
                )
            content = generate_completion(
                prompt=request.target_smiles,
                model=request.model,
            )
            return _reactant_plan_from_text(content, request.target_smiles)

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
            temperature=GENERATION_TEMPERATURE,
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
        temperature=REPAIR_TEMPERATURE,
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
            temperature=GENERATION_TEMPERATURE,
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
