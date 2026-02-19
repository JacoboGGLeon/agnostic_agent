from typing import Dict, Any, List
from agnostic_agent.core.contracts.embedding_provider import EmbeddingProvider
from agnostic_agent.app.errors import ProviderError
import json
import os

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

class BedrockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: Dict[str, Any]):
        if not HAS_BOTO3:
             raise ProviderError("boto3 package is required for BedrockEmbeddingProvider", provider="bedrock")

        self.config = config
        self.region_name = config.get("region_name") or os.getenv("AWS_REGION", "us-east-1")
        self.model = config.get("model", "amazon.titan-embed-text-v1")
        self._dimension = config.get("dimension", 1536) # Defaults to Titan v1 dim

        try:
            self.client = boto3.client("bedrock-runtime", region_name=self.region_name)
        except Exception as e:
            raise ProviderError(f"Failed to initialize Bedrock client: {e}", provider="bedrock")

    @property
    def dimension(self) -> int:
        return self._dimension

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
                 results.append(response_body.get("embedding"))
             except Exception as e:
                 raise ProviderError(f"Bedrock embedding failed for text snippet: {e}", provider="bedrock")
        return results
