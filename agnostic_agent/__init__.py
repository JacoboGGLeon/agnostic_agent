from __future__ import annotations

"""
Agnostic Deep Agent 2026 – paquete principal.

Arquitectura por módulos:

- setup.yaml        → configuración declarativa (modelos, memoria, planner, KBs, etc.).
- schemas.py        → contratos de datos internos (AnalyzerIntent, PlannerPlan, ToolRun…).
- capabilities.py   → capacidades de exploración de entornos:
                      · gestión de modelos / backends (vLLM, OpenAI, etc.)
                      · lanzamiento de servidores Qwen3+vLLM
                      · configuración del planner de herramientas.
- tools.py          → catálogo y registro de tools (toy, matemáticas, embeddings,
                      reranker, tools de contexto/tablas…).
- memory.py         → memoria de sesión / corto / largo plazo.
- prompts.py        → prompts por rol (analyzer, summarizer, validator, memory_write…).
- logic.py          → grafo maestro + sub-grafos
                      (ANALYZER, PLANNER, EXECUTOR, CATCHER, SUMMARIZER, VALIDATOR).
- agent.py          → clase Agent de alto nivel (init, run_turn).
- communication.py  → normalización de I/O (AgentInput, AgentOutput, vistas user/deep/dev).
- context.py        → definición de Knowledge Bases y conectores externos
                      (por ejemplo:
                       · BD tabular en SQLite
                       · VDB en sqlite-vec
                       · otras fuentes RAG, APIs, SQL, etc.).
"""

from .agent import Agent
from .communication import AgentInput, AgentOutput
from .capabilities import (
    PlannerConfig,
    QwenModelPaths,
    VllmConfig,
    VllmServers,
    VllmEndpoints,
    prepare_qwen_models,
    start_qwen_vllm_servers,
)
from .tools import get_default_tools
from .context import KnowledgeBase, get_default_context

__version__ = "0.2.0"

# Alias de compatibilidad con versiones anteriores (legacy)
AgentSession = Agent  # type: ignore

# API pública principal
__all__ = [
    # Núcleo del agente
    "Agent",
    "AgentSession",
    "AgentInput",
    "AgentOutput",
    # Configuración / modelos
    "PlannerConfig",
    "QwenModelPaths",
    "VllmConfig",
    "VllmServers",
    "VllmEndpoints",
    "prepare_qwen_models",
    "start_qwen_vllm_servers",
    # Contexto / tools de alto nivel
    "KnowledgeBase",
    "get_default_tools",
    "get_default_context",
    # Meta
    "__version__",
]
