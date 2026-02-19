from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    provider: str = "vllm"
    model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    base_url: Optional[str] = "http://localhost:8000/v1"
    api_key: Optional[str] = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 30

class EmbeddingConfig(BaseModel):
    provider: str = "vllm"
    model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    base_url: Optional[str] = "http://localhost:8000/v1"
    api_key: Optional[str] = "EMPTY"
    dimension: int = 3584

class VectorStoreConfig(BaseModel):
    provider: str = "sqlitevec"
    path: str = "embeddings.db"
    collection_name: str = "knowledge"

class PluginConfig(BaseModel):
    enabled: bool = True
    path: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)

class PluginsConfig(BaseModel):
    tools: Dict[str, PluginConfig] = Field(default_factory=dict)
    skills: Dict[str, PluginConfig] = Field(default_factory=dict)
    memory: Dict[str, PluginConfig] = Field(default_factory=dict)
    ui_panels: Dict[str, PluginConfig] = Field(default_factory=dict)

class AppConfig(BaseModel):
    environment: str = "dev"
    debug: bool = False
    log_level: str = "INFO"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
