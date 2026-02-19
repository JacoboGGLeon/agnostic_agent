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
            return response.choices[0].message.content
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
            return response.choices[0].message.content
        except Exception as e:
            raise ProviderError(f"vLLM chat failed: {e}", provider="vllm")
