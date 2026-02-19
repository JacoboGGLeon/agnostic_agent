from typing import Dict, Any, List
from agnostic_agent.core.contracts.embedding_provider import EmbeddingProvider
from agnostic_agent.app.errors import ProviderError
import openai
import os

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = config.get("model", "text-embedding-3-small")
        self._dimension = config.get("dimension", 1536)
        
        if not self.api_key:
             raise ProviderError("OpenAI API Key not found", provider="openai")
        self.client = openai.Client(api_key=self.api_key)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise ProviderError(f"OpenAI embedding failed: {e}", provider="openai")
