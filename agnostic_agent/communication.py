from __future__ import annotations

"""
Modelos de datos para la COMUNICACIÓN del Agnostic Deep Agent 2026.
COMPATIBILITY LAYER: Importa desde core.models.io_models
"""

from .core.models.io_models import (
    BaseAgentInput,
    AgentInput,
    ToolRun,
    AgentSummary,
    AgentView,
    AgentOutput,
)

__all__ = [
    "BaseAgentInput",
    "AgentInput",
    "ToolRun",
    "AgentSummary",
    "AgentView",
    "AgentOutput",
]
