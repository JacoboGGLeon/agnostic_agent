from typing import Dict, Any, Type
import importlib

from agnostic_agent.core.contracts.llm_provider import LLMProvider
from agnostic_agent.core.contracts.embedding_provider import EmbeddingProvider
from agnostic_agent.core.contracts.vectorstore_provider import VectorStoreProvider
from agnostic_agent.app.errors import ProviderError, ConfigurationError

class ProviderFactory:
    """
    Factory to instantiate providers based on configuration.
    Currently hardcoded/dynamic import mapping.
    """
    
    _llm_registry: Dict[str, str] = {
        "vllm": "agnostic_agent.providers.llm.vllm_provider.VLLMProvider",
        "openai": "agnostic_agent.providers.llm.openai_provider.OpenAIProvider",
        "bedrock": "agnostic_agent.providers.llm.bedrock_provider.BedrockProvider",
        # Adapters for legacy logic if needed
    }
    
    _embedding_registry: Dict[str, str] = {
        "vllm": "agnostic_agent.providers.embedding.vllm_provider.VLLMEmbeddingProvider",
        "openai": "agnostic_agent.providers.embedding.openai_provider.OpenAIEmbeddingProvider",
        "bedrock": "agnostic_agent.providers.embedding.bedrock_provider.BedrockEmbeddingProvider",
    }
    
    _vectorstore_registry: Dict[str, str] = {
        "sqlitevec": "agnostic_agent.providers.vectorstore.sqlitevec_provider.SQLiteVecProvider",
        "faiss": "agnostic_agent.providers.vectorstore.faiss_provider.FAISSProvider",
    }

    @classmethod
    def _import_class(cls, classpath: str) -> Type:
        module_name, class_name = classpath.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ProviderError(f"Failed to import provider class {classpath}: {e}", provider="unknown")

    @classmethod
    def get_llm_provider(cls, config: Dict[str, Any]) -> LLMProvider:
        provider_type = config.get("provider", "vllm")
        classpath = cls._llm_registry.get(provider_type)
        
        if not classpath:
            raise ConfigurationError(f"Unknown LLM provider type: {provider_type}")
            
        provider_class = cls._import_class(classpath)
        return provider_class(config)

    @classmethod
    def get_embedding_provider(cls, config: Dict[str, Any]) -> EmbeddingProvider:
        provider_type = config.get("provider", "vllm")
        classpath = cls._embedding_registry.get(provider_type)
        
        if not classpath:
            raise ConfigurationError(f"Unknown Embedding provider type: {provider_type}")
            
        provider_class = cls._import_class(classpath)
        return provider_class(config)

    @classmethod
    def get_vectorstore_provider(cls, config: Dict[str, Any]) -> VectorStoreProvider:
        provider_type = config.get("provider", "sqlitevec")
        classpath = cls._vectorstore_registry.get(provider_type)
        
        if not classpath:
            raise ConfigurationError(f"Unknown VectorStore provider type: {provider_type}")
            
        provider_class = cls._import_class(classpath)
        return provider_class(config)
