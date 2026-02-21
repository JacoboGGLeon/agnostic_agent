from typing import Dict, Any, List, Optional
from agnostic_agent.core.contracts.embedding_provider import EmbeddingProvider
from agnostic_agent.app.errors import ProviderError
import json
import os

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

class BedrockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: Dict[str, Any]):
        if not HAS_BOTO3:
             raise ProviderError("boto3 package is required for BedrockEmbeddingProvider", provider="bedrock")

        self.config = config
        self.region_name = _resolve_bedrock_region(config)
        self.model = config.get("model", "amazon.titan-embed-text-v1")
        self._dimension: Optional[int] = config.get("dimension")

        try:
            self.client = boto3.client("bedrock-runtime", region_name=self.region_name)
        except Exception as e:
            raise ProviderError(f"Failed to initialize Bedrock client: {e}", provider="bedrock")

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
             # Titan format
             body = json.dumps({"inputText": text})
             try:
                 response = self.client.invoke_model(
                     body=body,
                     modelId=self.model,
                     accept="application/json",
                     contentType="application/json"
                 )
                 response_body = json.loads(response.get("body").read())
                 emb = response_body.get("embedding")
                 if not isinstance(emb, list):
                     raise ProviderError(
                         f"Unexpected embedding response shape: {type(emb).__name__}",
                         provider="bedrock",
                     )
                 if self._dimension is None:
                     self._dimension = len(emb)
                 results.append(emb)
             except Exception as e:
                 raise ProviderError(f"Bedrock embedding failed for text snippet: {e}", provider="bedrock")
        return results
