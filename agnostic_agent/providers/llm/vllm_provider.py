from typing import Dict, Any, List, Optional, Generator
from agnostic_agent.core.contracts.llm_provider import LLMProvider
from agnostic_agent.app.errors import ProviderError
import openai

class VLLMProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "http://localhost:8000/v1")
        self.api_key = config.get("api_key", "EMPTY")
        self.model = config.get("model", "Qwen/Qwen2.5-7B-Instruct-AWQ")
        self.client = openai.Client(base_url=self.base_url, api_key=self.api_key)

    @staticmethod
    def _normalize_response(response: Any) -> Dict[str, Any]:
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        tool_calls = []
        for tc in (getattr(message, "tool_calls", None) or []):
            tool_calls.append(
                {
                    "id": getattr(tc, "id", None),
                    "name": getattr(getattr(tc, "function", None), "name", None),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                    "type": getattr(tc, "type", "function"),
                }
            )
        usage = {}
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": getattr(raw_usage, "prompt_tokens", None),
                "completion_tokens": getattr(raw_usage, "completion_tokens", None),
                "total_tokens": getattr(raw_usage, "total_tokens", None),
            }
        return {
            "text": getattr(message, "content", "") or "",
            "tool_calls": tool_calls,
            "usage": usage,
            "finish_reason": getattr(choice, "finish_reason", None),
            "raw": response,
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            normalized = self._normalize_response(response)
            return normalized["text"]
        except Exception as e:
            raise ProviderError(f"vLLM generation failed: {e}", provider="vllm")

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            raise ProviderError(f"vLLM stream failed: {e}", provider="vllm")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            normalized = self._normalize_response(response)
            return normalized["text"]
        except Exception as e:
            raise ProviderError(f"vLLM chat failed: {e}", provider="vllm")

    def chat_normalized(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            return self._normalize_response(response)
        except Exception as e:
            raise ProviderError(f"vLLM chat failed: {e}", provider="vllm")
