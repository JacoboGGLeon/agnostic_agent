from typing import Dict, Any, List
from agnostic_agent.core.contracts.embedding_provider import EmbeddingProvider
from agnostic_agent.app.errors import ProviderError
import openai

class VLLMEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "http://localhost:8000/v1")
        self.api_key = config.get("api_key", "EMPTY")
        self.model = config.get("model", "Qwen/Qwen2.5-7B-Instruct-AWQ")
        self._dimension = config.get("dimension", 3584)
        
        self.client = openai.Client(base_url=self.base_url, api_key=self.api_key)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Clean newlines which can mess up some tokenizers
        texts = [t.replace("\n", " ") for t in texts]
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise ProviderError(f"vLLM embedding failed: {e}", provider="vllm")
