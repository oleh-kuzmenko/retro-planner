import logging
from dataclasses import dataclass

from retro_planner.chemistry import canonicalize_smiles
from retro_planner.prompting import build_cot_prompt, build_cot_repair_prompt
from retro_planner.providers import LLMProvider
from retro_planner.reasoning import parse_reasoning_response, validate_precursors

LOGGER = logging.getLogger(__name__)

GENERATION_TEMPERATURE = 0.2
REPAIR_TEMPERATURE = 0.0


@dataclass(frozen=True)
class GenerationRequest:
    target_smiles: str
    llm_provider: LLMProvider
    model: str
    reactions: list[dict] | None = None
    temperature: float = GENERATION_TEMPERATURE


@dataclass(frozen=True)
class StepResult:
    think: str | None
    precursors: list[str]
    product_smiles: str
    raw_response: str
    warnings: list[str]
    errors: list[str]


def _call_provider(
    provider: LLMProvider,
    prompt: str,
    model: str,
    temperature: float,
) -> str:
    return provider.generate(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        json_mode=False,
    )


def generate_single_step(request: GenerationRequest) -> StepResult:
    """Run one single-step retrosynthetic generation: prompt -> LLM -> parse -> validate.

    One call = one retrosynthetic step (PZ section 2.4). Callers that want several
    candidate disconnections should call this repeatedly (e.g. varying temperature
    or RAG precedents), not ask the LLM for multiple routes in one response.
    """
    canonical_target = canonicalize_smiles(request.target_smiles) or request.target_smiles
    reactions = request.reactions or []
    prompt = build_cot_prompt(canonical_target, reactions)

    try:
        raw_response = _call_provider(
            request.llm_provider,
            prompt,
            request.model,
            request.temperature,
        )
    except Exception as exc:
        return StepResult(
            think=None,
            precursors=[],
            product_smiles=canonical_target,
            raw_response="",
            warnings=[],
            errors=[f"LLM API failure: {exc}"],
        )

    reasoning = parse_reasoning_response(raw_response)
    precursors, warnings, errors = validate_precursors(
        reasoning.answer_smiles,
        canonical_target,
    )

    if precursors is None:
        repair_prompt = build_cot_repair_prompt(
            canonical_target,
            reactions,
            reasoning.raw,
            errors,
        )
        try:
            repaired_raw = _call_provider(
                request.llm_provider,
                repair_prompt,
                request.model,
                REPAIR_TEMPERATURE,
            )
        except Exception as exc:
            warnings.append(f"Could not repair invalid response: {exc}")
            return StepResult(
                think=reasoning.think,
                precursors=[],
                product_smiles=canonical_target,
                raw_response=raw_response,
                warnings=warnings,
                errors=errors,
            )

        reasoning = parse_reasoning_response(repaired_raw)
        precursors, repair_warnings, errors = validate_precursors(
            reasoning.answer_smiles,
            canonical_target,
        )
        warnings.extend(repair_warnings)
        raw_response = repaired_raw

    return StepResult(
        think=reasoning.think,
        precursors=precursors or [],
        product_smiles=canonical_target,
        raw_response=raw_response,
        warnings=warnings,
        errors=errors,
    )
