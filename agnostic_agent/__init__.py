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
- context.py        → (DEPRECATED -> see knowledge/) definición de Knowledge Bases.
- knowledge/        → Nuevo sistema modular de conocimiento (vector, sql, tabular, json).
"""

from .agent import Agent
from .communication import AgentInput, AgentOutput
from .capabilities import (
    PlannerConfig,
    LocalModelPaths,
    VllmConfig,
    VllmServers,
    VllmEndpoints,
    prepare_local_models,
    start_local_vllm_servers,
)
from .tools import get_default_tools
from .knowledge import KnowledgeBase, get_default_context
from .skills import SkillRegistry, Skill

from .legacy.compatibility import AgentSession

__version__ = "0.3.0"  # Bump version for V2 release

# API pública principal
__all__ = [
    # Núcleo del agente
    "Agent",
    "AgentSession",
    "AgentInput",
    "AgentOutput",
    # Configuración / modelos
    "PlannerConfig",
    "LocalModelPaths",
    "VllmConfig",
    "VllmServers",
    "VllmEndpoints",
    "prepare_local_models",
    "start_local_vllm_servers",
    # Contexto / tools de alto nivel
    "KnowledgeBase",
    "get_default_tools",
    "get_default_context",
    "SkillRegistry",
    "Skill",
    # Meta
    "__version__",
]
