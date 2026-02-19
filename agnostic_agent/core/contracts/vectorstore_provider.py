from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStoreProvider(ABC):
    """
    Abstract base class for VectorStore providers.
    """
    
    @abstractmethod
    def add(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the vector store."""
        pass
        
    @abstractmethod
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        Returns a list of dicts with 'content', 'metadata', 'score'.
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        pass
