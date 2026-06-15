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
    key_url: str
    create_provider: Callable[[str], LLMProvider]


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


LLM_PROVIDER_REGISTRY: dict[str, LLMProviderConfig] = {
    "groq": LLMProviderConfig(
        key="groq",
        label="Groq",
        api_key_env_var="GROQ_API_KEY",
        model_env_var="GROQ_MODEL",
        default_model="llama-3.3-70b-versatile",
        key_url="https://console.groq.com/keys",
        create_provider=GroqLLMProvider,
    ),
}
