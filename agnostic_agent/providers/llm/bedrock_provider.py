from typing import Dict, Any, List, Optional, Generator
from agnostic_agent.core.contracts.llm_provider import LLMProvider
from agnostic_agent.app.errors import ProviderError
import json
import os

# Try to import boto3, but don't crash if missing (soft dependency)
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

class BedrockProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        if not HAS_BOTO3:
            raise ProviderError("boto3 package is required for BedrockProvider", provider="bedrock")
        
        self.config = config
        self.region_name = config.get("region_name") or os.getenv("AWS_REGION", "us-east-1")
        self.model = config.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
        
        try:
            self.client = boto3.client("bedrock-runtime", region_name=self.region_name)
        except Exception as e:
            raise ProviderError(f"Failed to initialize Bedrock client: {e}", provider="bedrock")

    def _prepare_body(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Generic body for Claude 3 (most common bedrock use case here)
        # TODO: Add logic for other models (Titan, Llama) if needed
        serialized_messages = []
        system_prompt = None
        
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system_prompt = content
                continue
            serialized_messages.append({"role": role, "content": [{"type": "text", "text": content}]})

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get("max_tokens", 2048),
            "messages": serialized_messages,
            "temperature": kwargs.get("temperature", 0.0)
        }
        if system_prompt:
            body["system"] = system_prompt
            
        return json.dumps(body)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, **kwargs)

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
         messages = []
         if system_prompt:
             messages.append({"role": "system", "content": system_prompt})
         messages.append({"role": "user", "content": prompt})
         
         # Stub: Bedrock streaming logic is complex, skipping strict implementation for MVP
         # Just return full response as one chunk for now to avoid complexity without boto3 env
         yield self.chat(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        body = self._prepare_body(messages, **kwargs)
        try:
            response = self.client.invoke_model(
                body=body,
                modelId=self.model,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("content", [])[0].get("text", "")
        except Exception as e:
             raise ProviderError(f"Bedrock chat failed: {e}", provider="bedrock")
