from types import SimpleNamespace

from retro_eval.providers.chat_api import OpenAICompatibleLLMProvider


class _StubCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = SimpleNamespace(content="<answer>CCO</answer>")
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None)


def _provider_with_stub() -> tuple[OpenAICompatibleLLMProvider, _StubCompletions]:
    provider = OpenAICompatibleLLMProvider(api_key="test", base_url="https://example.invalid/v1")
    stub = _StubCompletions()
    provider.client.chat.completions = stub
    return provider, stub


def test_generate_omits_reasoning_effort_by_default():
    provider, stub = _provider_with_stub()

    provider.generate(messages=[{"role": "user", "content": "hi"}], model="llama-3.3-70b-versatile", temperature=0.0)

    assert "reasoning_effort" not in stub.last_kwargs


def test_generate_forwards_reasoning_effort_when_set():
    provider, stub = _provider_with_stub()

    provider.generate(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-oss-120b",
        temperature=0.0,
        reasoning_effort="high",
    )

    assert stub.last_kwargs["reasoning_effort"] == "high"
