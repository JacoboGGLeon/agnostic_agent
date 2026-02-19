from typing import Dict, Any, List, Optional
from agnostic_agent.core.contracts.vectorstore_provider import VectorStoreProvider
from agnostic_agent.app.errors import ProviderError
import numpy as np
import pickle
import os

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class FAISSProvider(VectorStoreProvider):
    def __init__(self, config: Dict[str, Any]):
        if not HAS_FAISS:
             # Soft fail? Or warn. Typically raise if requested specifically.
             # We assume if this class is instantiated, the user wants FAISS.
             raise ProviderError("faiss package is required for FAISSProvider", provider="faiss")

        self.config = config
        self.index_path = config.get("path", "faiss_index.bin")
        self.dimension = config.get("dimension") # Should be provided or inferred on first add
        
        self.index = None
        self.doc_store = {} # ID -> {content, metadata}
        
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                # Load docstore
                store_path = self.index_path + ".store"
                if os.path.exists(store_path):
                    with open(store_path, "rb") as f:
                        self.doc_store = pickle.load(f)
            except Exception as e:
                raise ProviderError(f"Failed to load FAISS index: {e}", provider="faiss")

    def _save(self):
        if self.index:
            try:
                faiss.write_index(self.index, self.index_path)
                with open(self.index_path + ".store", "wb") as f:
                    pickle.dump(self.doc_store, f)
            except Exception as e:
                 raise ProviderError(f"Failed to save FAISS index: {e}", provider="faiss")

    def add(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        if not embeddings: return []
        
        dim = len(embeddings[0])
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        
        # Convert to float32 numpy array
        emb_np = np.array(embeddings).astype('float32')
        self.index.add(emb_np)
        
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        # Store metadata (Naive mapping strategies, for real production use generic ID mapping)
        # Here we just append to store assuming sequential ID if we were using IndexIDMap
        # But IndexFlatL2 doesn't support IDs directly.
        # We need to manage mapping index_id (0..N) to doc_id. 
        # For this MVP, we will simplify and assume we can reload and rebuild or just use the doc_store keys as 0..N
        # This is non-trivial for persistent FAISS updates without IndexIDMap2.
        
        # Proper way: IndexIDMap
        # For now, we'll store in doc_store with keys being the new indices.
        # This is a limitation: deletion is hard.
        
        start_idx = self.index.ntotal - len(texts)
        for i, text in enumerate(texts):
            idx = start_idx + i
            self.doc_store[idx] = {
                "id": ids[i],
                "content": text,
                "metadata": metadatas[i]
            }
            
        self._save()
        return ids

    def search(
        self, 
        query_embedding: List[float], 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not self.index or self.index.ntotal == 0:
            return []
            
        q_np = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(q_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            if idx in self.doc_store:
                doc = self.doc_store[idx]
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(1.0 / (1.0 + distances[0][i])) # Convert L2 distance to rough score
                })
        
        return results

    def delete(self, ids: List[str]) -> None:
        # Not supported in simple IndexFlatL2 without rebuild
        pass
