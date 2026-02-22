import pytest
import os
from agnostic_agent.providers.vectorstore.sqlitevec_provider import SQLiteVecProvider

@pytest.fixture
def sqlite_provider(tmp_path):
    db_path = tmp_path / "test.db"
    config = {
        "provider": "sqlitevec",
        "path": str(db_path),
        "collection_name": "test_collection"
    }
    return SQLiteVecProvider(config)

def test_add_and_search(sqlite_provider):
    texts = ["apple", "banana", "orange"]
    # Dummy embeddings (3D)
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    metadatas = [{"type": "fruit"}, {"type": "fruit"}, {"type": "fruit"}]
    
    ids = sqlite_provider.add(texts, embeddings, metadatas)
    assert len(ids) == 3
    
    # Search for something close to apple
    query = [0.9, 0.1, 0.0] 
    results = sqlite_provider.search(query, k=1)
    
    assert len(results) == 1
    assert results[0]["content"] == "apple"
    assert results[0]["score"] > 0.8
    
def test_delete(sqlite_provider):
    texts = ["to_delete"]
    embeddings = [[0.5, 0.5, 0.0]]
    ids = sqlite_provider.add(texts, embeddings)
    
    assert len(sqlite_provider.search(embeddings[0], k=1)) == 1
    
    sqlite_provider.delete(ids)
    
    results = sqlite_provider.search(embeddings[0], k=1)
    # Depending on implementation, might return empty or low score if db empty
    # Here DB is not empty, but "to_delete" should be gone.
    # Note: Search returns whatever is there, so verify by ID or content
    found = [r for r in results if r["id"] == ids[0]]
    assert len(found) == 0
