"""Hosted OpenAI-compatible chat-API provider: Groq, Cerebras, OpenRouter,
Together, Fireworks, Ollama, llama.cpp server, OpenAI itself, or any other
OpenAI-compatible endpoint, selected via `--base-url`.

Used directly by `scripts/models/run_rag_cot_llm.py`/`run_cot_llm.py` and
`scripts/models/run_chat_zero_shot.py`/`run_qwen_lora_peft.py`'s Ollama backend.
"""

import json
import logging
import time

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
        json_mode: bool = False,
        reasoning_effort: str | None = None,
    ) -> str:
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            request["reasoning_effort"] = reasoning_effort

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
