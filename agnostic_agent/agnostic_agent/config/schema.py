"""
Compatibility wrapper.

Canonical config schema lives in `agnostic_agent.config.schema`.
This module is kept only to avoid breaking old imports.
"""

from agnostic_agent.config.schema import (  # noqa: F401
    AppConfig,
    EmbeddingConfig,
    LLMConfig,
    PluginConfig,
    PluginsConfig,
    VectorStoreConfig,
)
