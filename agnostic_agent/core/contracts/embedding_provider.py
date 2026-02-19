from abc import ABC, abstractmethod
from typing import List, Union

class EmbeddingProvider(ABC):
    """
    Abstract base class for Embedding providers.
    """
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        pass
        
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass
        
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass
