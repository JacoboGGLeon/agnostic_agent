from types import SimpleNamespace

from agnostic_agent.providers.llm.openai_provider import OpenAIProvider
from agnostic_agent.providers.llm.vllm_provider import VLLMProvider


def _fake_completion():
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="search", arguments='{"q":"test"}'),
    )
    message = SimpleNamespace(content="hello", tool_calls=[tool_call])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    return SimpleNamespace(choices=[choice], usage=usage)


def _patched_client(monkeypatch, module_path):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: _fake_completion(),
            )
        )
    )
    monkeypatch.setattr(module_path, lambda **kwargs: fake_client)


def _assert_normalized_shape(data):
    assert set(data.keys()) == {"text", "tool_calls", "usage", "finish_reason", "raw"}
    assert isinstance(data["text"], str)
    assert isinstance(data["tool_calls"], list)
    assert isinstance(data["usage"], dict)
    assert "total_tokens" in data["usage"]


def test_openai_provider_chat_normalized(monkeypatch):
    _patched_client(monkeypatch, "agnostic_agent.providers.llm.openai_provider.openai.Client")
    provider = OpenAIProvider({"api_key": "sk-test", "model": "gpt-4o-mini"})
    out = provider.chat_normalized([{"role": "user", "content": "hi"}])
    _assert_normalized_shape(out)
    assert out["text"] == "hello"
    assert out["tool_calls"][0]["name"] == "search"


def test_vllm_provider_chat_normalized(monkeypatch):
    _patched_client(monkeypatch, "agnostic_agent.providers.llm.vllm_provider.openai.Client")
    provider = VLLMProvider({"base_url": "http://localhost:8000/v1", "api_key": "EMPTY"})
    out = provider.chat_normalized([{"role": "user", "content": "hi"}])
    _assert_normalized_shape(out)
    assert out["finish_reason"] == "stop"
