import pytest
from unittest.mock import MagicMock, patch
from agnostic_agent.providers.factory import ProviderFactory
from agnostic_agent.app.errors import ConfigurationError
from agnostic_agent.core.contracts.llm_provider import LLMProvider

def test_get_llm_provider_vllm():
    config = {"provider": "vllm", "base_url": "http://test"}
    # We mock the import to avoid actual dependency requirements during unit test
    with patch("agnostic_agent.providers.factory.importlib.import_module") as mock_import:
        mock_class = MagicMock()
        mock_module = MagicMock()
        setattr(mock_module, "VLLMProvider", mock_class)
        mock_import.return_value = mock_module
        
        provider = ProviderFactory.get_llm_provider(config)
        
        mock_import.assert_called_with("agnostic_agent.providers.llm.vllm_provider")
        mock_class.assert_called_with(config)

def test_get_llm_provider_unknown():
    config = {"provider": "unknown_provider"}
    with pytest.raises(ConfigurationError):
        ProviderFactory.get_llm_provider(config)

def test_get_embedding_provider_openai():
    config = {"provider": "openai", "api_key": "sk-test"}
    with patch("agnostic_agent.providers.factory.importlib.import_module") as mock_import:
        mock_class = MagicMock()
        mock_module = MagicMock()
        setattr(mock_module, "OpenAIEmbeddingProvider", mock_class)
        mock_import.return_value = mock_module
        
        provider = ProviderFactory.get_embedding_provider(config)
        
        mock_import.assert_called_with("agnostic_agent.providers.embedding.openai_provider")

def test_get_vectorstore_provider_sqlitevec():
    config = {"provider": "sqlitevec"}
    with patch("agnostic_agent.providers.factory.importlib.import_module") as mock_import:
        mock_class = MagicMock()
        mock_module = MagicMock()
        setattr(mock_module, "SQLiteVecProvider", mock_class)
        mock_import.return_value = mock_module
        
        provider = ProviderFactory.get_vectorstore_provider(config)
        
        mock_import.assert_called_with("agnostic_agent.providers.vectorstore.sqlitevec_provider")
