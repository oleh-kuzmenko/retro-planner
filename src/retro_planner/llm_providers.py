import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


LOGGER = logging.getLogger(__name__)


def _json_log(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    return json.dumps(value, indent=2, default=str, ensure_ascii=True)


def _response_log(content: str) -> str:
    try:
        return _json_log(json.loads(content))
    except json.JSONDecodeError:
        return _json_log({"content": content})


class LLMProvider(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        json_mode: bool = True,
    ) -> str:
        """Return raw model text for the supplied chat messages."""
        ...


@dataclass(frozen=True)
class LLMProviderConfig:
    key: str
    label: str
    api_key_env_var: str
    model_env_var: str
    default_model: str
    create_provider: Callable[[str, str | None], LLMProvider]
    key_url: str | None = None
    api_key_required: bool = True
    base_url_env_var: str | None = None
    default_base_url: str | None = None


REACTION_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "reaction_name": {"type": "string"},
        "reactants": {
            "type": "array",
            "items": {"type": "string"},
        },
        "product_smiles": {"type": "string"},
        "stoichiometry": {"type": "string"},
        "reagents": {"type": "string"},
        "solvent": {"type": "string"},
        "temperature_celsius": {"type": "string"},
        "reaction_time": {"type": "string"},
        "atmosphere": {"type": "string"},
        "workup_purification": {"type": "string"},
        "expected_yield_percent": {"type": "string"},
        "important_conditions": {"type": "string"},
        "rationale": {"type": "string"},
        "objective_fit": {"type": "string"},
        "evidence_reaction_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "additionalProperties": True,
}


RETROSYNTHESIS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "routes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "route_name": {"type": "string"},
                    "summary": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": REACTION_STEP_SCHEMA,
                    },
                    "objective_fit": {"type": "string"},
                    "evidence_reaction_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["steps"],
                "additionalProperties": True,
            },
        },
        "overall_recommendation": {"type": "string"},
    },
    "required": ["routes"],
    "additionalProperties": True,
}


class GroqLLMProvider:
    def __init__(self, api_key: str):
        from groq import Groq

        self.client = Groq(api_key=api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        json_mode: bool = True,
    ) -> str:
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        LOGGER.info(
            "Groq request started model=%s temperature=%.2f json_mode=%s messages=%d",
            model,
            temperature,
            json_mode,
            len(messages),
        )
        LOGGER.info("Groq request payload:\n%s", _json_log(request))
        started_at = time.perf_counter()

        try:
            completion = self.client.chat.completions.create(**request)
        except Exception:
            LOGGER.exception(
                "Groq request failed model=%s duration_seconds=%.3f",
                model,
                time.perf_counter() - started_at,
            )
            raise

        content = completion.choices[0].message.content or ""
        choice = completion.choices[0]
        usage = getattr(completion, "usage", None)
        LOGGER.info(
            "Groq response received model=%s duration_seconds=%.3f "
            "finish_reason=%s response_chars=%d usage=%s",
            model,
            time.perf_counter() - started_at,
            getattr(choice, "finish_reason", None),
            len(content),
            _json_log(usage) if usage is not None else "null",
        )
        LOGGER.info("Groq response payload:\n%s", _response_log(content))
        return content


class OpenAILLMProvider:
    def __init__(self, api_key: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        json_mode: bool = True,
    ) -> str:
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        completion = self.client.chat.completions.create(**request)
        return completion.choices[0].message.content or ""


class OpenAICompatibleLLMProvider:
    def __init__(self, api_key: str, base_url: str | None):
        from openai import OpenAI

        if not base_url:
            raise ValueError("Custom LLM base URL is required.")

        self.client = OpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url.rstrip("/"),
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        json_mode: bool = True,
    ) -> str:
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "retrosynthesis_plan",
                    "schema": RETROSYNTHESIS_RESPONSE_SCHEMA,
                },
            }

        LOGGER.info(
            "Custom OpenAI chat request started model=%s temperature=%.2f "
            "json_mode=%s messages=%d",
            model,
            temperature,
            json_mode,
            len(messages),
        )
        LOGGER.info("Custom OpenAI chat request payload:\n%s", _json_log(request))
        started_at = time.perf_counter()

        try:
            completion = self.client.chat.completions.create(**request)
        except Exception:
            LOGGER.exception(
                "Custom OpenAI chat request failed model=%s duration_seconds=%.3f",
                model,
                time.perf_counter() - started_at,
            )
            raise

        choice = completion.choices[0]
        content = choice.message.content or ""
        usage = getattr(completion, "usage", None)
        LOGGER.info(
            "Custom OpenAI chat response received model=%s duration_seconds=%.3f "
            "finish_reason=%s response_chars=%d usage=%s",
            model,
            time.perf_counter() - started_at,
            getattr(choice, "finish_reason", None),
            len(content),
            _json_log(usage) if usage is not None else "null",
        )
        LOGGER.info("Custom OpenAI chat model response:\n%s", content)
        return content

    def generate_completion(
        self,
        prompt: str,
        model: str,
    ) -> str:
        request = {
            "model": model,
            "prompt": prompt,
        }
        LOGGER.info(
            "Custom OpenAI completion request started model=%s prompt_chars=%d",
            model,
            len(prompt),
        )
        LOGGER.info(
            "Custom OpenAI completion request payload:\n%s",
            _json_log(request),
        )
        started_at = time.perf_counter()

        try:
            completion = self.client.completions.create(**request)
        except Exception:
            LOGGER.exception(
                "Custom OpenAI completion request failed model=%s "
                "duration_seconds=%.3f",
                model,
                time.perf_counter() - started_at,
            )
            raise

        choice = completion.choices[0]
        content = choice.text or ""
        usage = getattr(completion, "usage", None)
        LOGGER.info(
            "Custom OpenAI completion response received model=%s "
            "duration_seconds=%.3f finish_reason=%s response_chars=%d usage=%s",
            model,
            time.perf_counter() - started_at,
            getattr(choice, "finish_reason", None),
            len(content),
            _json_log(usage) if usage is not None else "null",
        )
        LOGGER.info("Custom OpenAI completion model response:\n%s", content)
        return content


LLM_PROVIDER_REGISTRY: dict[str, LLMProviderConfig] = {
    "groq": LLMProviderConfig(
        key="groq",
        label="Groq",
        api_key_env_var="GROQ_API_KEY",
        model_env_var="GROQ_MODEL",
        default_model="llama-3.3-70b-versatile",
        create_provider=lambda api_key, _base_url: GroqLLMProvider(api_key),
        key_url="https://console.groq.com/keys",
    ),
    "openai": LLMProviderConfig(
        key="openai",
        label="OpenAI",
        api_key_env_var="OPENAI_API_KEY",
        model_env_var="OPENAI_MODEL",
        default_model="gpt-4.1",
        create_provider=lambda api_key, _base_url: OpenAILLMProvider(api_key),
        key_url="https://platform.openai.com/api-keys",
    ),
    "custom_openai": LLMProviderConfig(
        key="custom_openai",
        label="Custom / OpenAI-compatible",
        api_key_env_var="CUSTOM_LLM_API_KEY",
        model_env_var="CUSTOM_LLM_MODEL",
        default_model="llama3.1",
        create_provider=OpenAICompatibleLLMProvider,
        api_key_required=False,
        base_url_env_var="CUSTOM_LLM_BASE_URL",
        default_base_url="http://localhost:11434/v1",
    ),
}
