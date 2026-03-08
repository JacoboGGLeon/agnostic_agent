from typing import Dict, Any, List, Optional, Generator, Tuple
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

def _resolve_bedrock_region(config: Dict[str, Any], default_region: str = "us-east-1") -> str:
    if not HAS_BOTO3:
        return (
            config.get("region_name")
            or config.get("region")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or default_region
        )
    session_region = boto3.session.Session().region_name
    return (
        config.get("region_name")
        or config.get("region")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or session_region
        or default_region
    )

def _is_anthropic_model(model_id: str) -> bool:
    mid = (model_id or "").strip().lower()
    return mid.startswith("anthropic.") or "claude" in mid

def _extract_system_and_messages(
    messages: List[Dict[str, str]],
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    system_prompt: Optional[str] = None
    normalized: List[Dict[str, str]] = []
    for m in messages or []:
        role = (m.get("role") or "").strip().lower()
        content = m.get("content") or ""
        if role == "system":
            system_prompt = str(content)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        normalized.append({"role": role, "content": str(content)})
    return system_prompt, normalized

class BedrockProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        if not HAS_BOTO3:
            raise ProviderError("boto3 package is required for BedrockProvider", provider="bedrock")
        
        self.config = config
        self.region_name = _resolve_bedrock_region(config)
        self.model = config.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
        self.api = (config.get("api") or os.getenv("BEDROCK_API") or "").strip().lower() or "auto"
        
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

    def _chat_invoke_model(self, messages: List[Dict[str, str]], **kwargs) -> str:
        body = self._prepare_body(messages, **kwargs)
        try:
            response = self.client.invoke_model(
                body=body,
                modelId=self.model,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("content", [])[0].get("text", "")
        except Exception as e:
            raise ProviderError(f"Bedrock invoke_model chat failed: {e}", provider="bedrock")

    def _chat_converse(self, messages: List[Dict[str, str]], **kwargs) -> str:
        system_prompt, norm_msgs = _extract_system_and_messages(messages)

        bedrock_messages = [
            {"role": m["role"], "content": [{"text": m["content"]}]} for m in norm_msgs
        ]

        # Optional inference config (best-effort; not all models support all fields).
        inference_cfg: Dict[str, Any] = {}
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            inference_cfg["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            inference_cfg["maxTokens"] = kwargs["max_tokens"]

        # Prefer passing system prompt explicitly; if the SDK/model rejects it, fallback gracefully.
        call_kwargs: Dict[str, Any] = {}
        if inference_cfg:
            call_kwargs["inferenceConfig"] = inference_cfg

        try:
            if system_prompt:
                try:
                    response = self.client.converse(
                        modelId=self.model,
                        system=[{"text": system_prompt}],
                        messages=bedrock_messages,
                        **call_kwargs,
                    )
                except TypeError:
                    # Some SDK versions/models may not support `system=`.
                    bedrock_messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": f"{system_prompt}\n\n---\n\n"
                                    + (bedrock_messages[0]["content"][0]["text"] if bedrock_messages else ""),
                                }
                            ],
                        }
                    ] + (bedrock_messages[1:] if len(bedrock_messages) > 1 else [])
                    response = self.client.converse(
                        modelId=self.model,
                        messages=bedrock_messages,
                        **call_kwargs,
                    )
            else:
                response = self.client.converse(
                    modelId=self.model,
                    messages=bedrock_messages,
                    **call_kwargs,
                )
        except TypeError:
            # Last resort: drop inference config if not supported.
            if system_prompt:
                response = self.client.converse(
                    modelId=self.model,
                    system=[{"text": system_prompt}],
                    messages=bedrock_messages,
                )
            else:
                response = self.client.converse(
                    modelId=self.model,
                    messages=bedrock_messages,
                )
        except Exception as e:
            raise ProviderError(f"Bedrock converse chat failed: {e}", provider="bedrock")

        try:
            content = response["output"]["message"]["content"]
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        except Exception as e:
            raise ProviderError(f"Bedrock converse response parse failed: {e}", provider="bedrock")

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
        api = self.api
        if api == "auto":
            api = "invoke_model" if _is_anthropic_model(self.model) else "converse"

        if api == "invoke_model":
            return self._chat_invoke_model(messages, **kwargs)
        if api == "converse":
            return self._chat_converse(messages, **kwargs)

        raise ProviderError(f"Unknown Bedrock API mode: {self.api}", provider="bedrock")

    def chat_normalized(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        text = self.chat(messages, **kwargs)
        return {
            "text": text or "",
            "tool_calls": [],
            "usage": {},
            "finish_reason": None,
            "raw": {"provider": "bedrock", "api": self.api, "model": self.model},
        }
