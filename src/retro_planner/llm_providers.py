from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


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

        completion = self.client.chat.completions.create(**request)
        return completion.choices[0].message.content or ""


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
            base_url=base_url,
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
            request["response_format"] = {"type": "json_object"}

        completion = self.client.chat.completions.create(**request)
        return completion.choices[0].message.content or ""


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
