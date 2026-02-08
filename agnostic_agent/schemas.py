from __future__ import annotations

"""
Esquemas y Contratos del Agnostic Deep Agent 2026.

ARQUITECTURA DE CONTRATOS - SEPARACIÓN DE RESPONSABILIDADES:

┌─────────────────────────────────────────────────────────────────┐
│                     SEPARACIÓN DE RESPONSABILIDADES              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📥 communication.py  → I/O EXTERNO (API, Streamlit, Usuarios)  │
│      - AgentInput      : Lo que recibe el agente                │
│      - AgentOutput     : Lo que devuelve el agente              │
│      - AgentView       : Vistas por rol (dev/deep/user)         │
│      - AgentSummary    : Resumen del pipeline (externo)         │
│      - ToolRun         : Ejecución de herramienta (externo)     │
│                                                                  │
│  ⚙️  logic.py         → GRAFO INTERNO (LangGraph State)         │
│      - State           : Estado compartido del grafo            │
│      - AnalyzerResult  : Resultado del nodo Analyzer            │
│      - PlannerTrajectory: Trazas de planificación               │
│      - ExecutorStep    : Pasos ejecutados                       │
│      - SummaryDict     : Resumen interno del pipeline           │
│      - ValidatorResult : Resultado de validación                │
│                                                                  │
│  📦 schemas.py        → RE-EXPORTACIÓN + DOCUMENTACIÓN (ESTE)   │
│      - Re-exporta tipos de communication.py y logic.py          │
│      - Punto centralizado de importación                        │
│      - Documentación de la arquitectura                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

REGLAS:
1. NO duplicar modelos entre archivos
2. Importar desde el archivo de origen según responsabilidad
3. Este archivo (schemas.py) es solo para re-exportación y documentación

EJEMPLO DE USO:
    # ✅ Correcto - Importar desde el origen
    from agnostic_agent.communication import AgentInput, AgentOutput
    from agnostic_agent.logic import State, AnalyzerResult
    
    # ✅ También correcto - Importar desde schemas (re-exportación)
    from agnostic_agent.schemas import AgentInput, State
    
    # ❌ Incorrecto - Redefinir modelos ya existentes
    # class AgentInput(BaseModel): ...  # ¡NO!
"""

# Re-exportaciones de communication.py (I/O Externo)
from .communication import (
    AgentInput,
    BaseAgentInput,
    AgentOutput,
    AgentView,
    AgentSummary,
    ToolRun,
)

# Re-exportaciones de logic.py (Grafo Interno)
from .logic import (
    State,
    AnalyzerResult,
    PlannerTrajectory,
    ExecutorStep,
    SummaryDict,
    ValidatorResult,
)

# Re-exportaciones de knowledge
from .knowledge.types import (
    KnowledgeBase,
    ElementNode,
    Chunk,
)

# Exponer todo para facilitar imports
__all__ = [
    # I/O Externo
    "AgentInput",
    "BaseAgentInput",
    "AgentOutput",
    "AgentView",
    "AgentSummary",
    "ToolRun",
    # Grafo Interno
    "State",
    "AnalyzerResult",
    "PlannerTrajectory",
    "ExecutorStep",
    "SummaryDict",
    "ValidatorResult",
    # Knowledge
    "KnowledgeBase",
    "ElementNode",
    "Chunk",
]
