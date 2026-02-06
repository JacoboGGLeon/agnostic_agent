from __future__ import annotations

"""
Esquemas internos del Agnostic Deep Agent 2026.

Aquí definimos el "lenguaje interno" del grafo:

- AnalyzerIntent      → resultado del sub-grafo ANALYZER.
- PlannerPlan         → plan de pasos del sub-grafo PLANNER.
- ValidationResult    → veredicto del sub-grafo VALIDATOR.
- MemoryContext       → contexto recuperado por MEMORY_READ.
- AgentState          → TypedDict compartido por todos los nodos del grafo.

NOTA:
- Estos modelos NO son el I/O externo; para eso está `communication.py`.
"""

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

from .communication import ToolRun, AgentSummary


# ─────────────────────────────────────────────
# ANALYZER
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# ANALYZER
# ─────────────────────────────────────────────

class Proposition(BaseModel):
    """
    Unidad atómica de instrucción extraída del prompt del usuario.
    """
    id: str = Field(..., description="ID único de la proposición (ej. 'P1').")
    text: str = Field(..., description="Texto de la instrucción atómica.")
    confidence: float = Field(default=1.0, description="Nivel de confianza (0.0 a 1.0).")


class RequiredItem(BaseModel):
    """
    Ítem que DEBE quedar cubierto en la respuesta final.
    """
    id: str = Field(..., description="Identificador corto del ítem requerido.")
    description: str = Field(..., description="Descripción del ítem requerido.")
    must_be_answered: bool = Field(
        default=True,
        description="Si True, la respuesta final DEBE cubrir este ítem.",
    )


class AnalyzerIntent(BaseModel):
    """
    Resultado del ANALYZER:

    - propositions: Lista de proposiciones atómicas extraídas.
    - logic_form: (Opcional) Representación lógica global.
    - main_objective: Objetivo principal inferido.
    - language: Idioma dominante del usuario.
    """
    propositions: List[Proposition] = Field(default_factory=list)
    logic_form: str = ""
    main_objective: str = ""
    language: Optional[str] = None
    reasoning_content: Optional[str] = None
    
    # Legacy/Compatibilidad (si se necesita)
    subqueries: List[str] = Field(default_factory=list)
    required_items: List[RequiredItem] = Field(default_factory=list)
    wants_tool_trace: bool = False


# ─────────────────────────────────────────────
# PLANNER
# ─────────────────────────────────────────────

from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class TaskNode(BaseModel):
    """
    Nodo del Árbol de Tareas (Task Tree).
    
    Representa un paso ejecutable en el plan.
    """
    id: str = Field(..., description="ID único de la tarea (ej. 'T1').")
    instruction: str = Field(..., description="Instrucción de lo que debe hacer esta tarea.")
    dependencies: List[str] = Field(default_factory=list, description="IDs de tareas que deben completarse antes.")
    
    # Estado de ejecución
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    result: Optional[Any] = None
    error: Optional[str] = None

    # Definición de ejecución (Tool call)
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata traza (opcional)
    thought: Optional[str] = None


class PlannerPlan(BaseModel):
    """
    Plan completo devuelto por el PLANNER como DAG de TaskNode.
    """
    tasks: List[TaskNode] = Field(default_factory=list, description="Lista plana de nodos que forman el DAG.")
    rationale: Optional[str] = None
    reasoning_content: Optional[str] = None


# ─────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────

class ValidationResult(BaseModel):
    """
    Resultado del VALIDATOR:

    - all_covered: True si todos los RequiredItem se consideran cubiertos.
    - missing_item_ids: ids de RequiredItem que faltan por cubrir.
    - comments: texto adicional (por ejemplo, sugerencias al SUMMARIZER).
    """
    all_covered: bool
    missing_item_ids: List[str] = Field(default_factory=list)
    comments: Optional[str] = None


# ─────────────────────────────────────────────
# MEMORY
# ─────────────────────────────────────────────

class MemoryContext(BaseModel):
    """
    Contexto de memoria agregado por MEMORY_READ.

    - session_history: extractos de la conversación actual.
    - short_term_snippets: resúmenes o puntos clave recientes.
    - long_term_snippets: recuerdos persistentes relevantes (vector store, etc.).
    """
    session_history: List[str] = Field(default_factory=list)
    short_term_snippets: List[str] = Field(default_factory=list)
    long_term_snippets: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# STATE GLOBAL (LangGraph)
# ─────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    Estado compartido entre nodos del grafo.

    Claves más relevantes:
    - user_prompt: texto original del usuario (último turno).
    - session_id, user_id: identificadores lógicos.
    - setup_path: ruta al setup.yaml activo.
    - kb_names: nombres de KBs solicitadas por el usuario.

    - memory_context: MemoryContext completado por MEMORY_READ.
    - analyzer_intent: AnalyzerIntent generado por ANALYZER.
    - planner_plan: PlannerPlan generado por PLANNER.

    - step_results: resultados crudos por id de step.
    - tool_runs: lista de ToolRun (normalizados en CATCHER).
    - validation: ValidationResult del VALIDATOR.

    - pipeline_summary: AgentSummary opcional con resumen por fase.
    - user_out / deep_out / dev_out: textos finales por vista.
    """
    # Identificadores y config
    user_prompt: str
    session_id: str
    user_id: Optional[str]
    setup_path: str
    kb_names: List[str]

    # Contexto de memoria
    memory_context: MemoryContext

    # Resultados de sub-grafos
    analyzer_intent: AnalyzerIntent
    planner_plan: PlannerPlan
    validation: ValidationResult

    # Ejecución de tools
    step_results: Dict[str, Any]
    tool_runs: List[ToolRun]

    # Resúmenes y salidas finales
    pipeline_summary: AgentSummary
    user_out: str
    deep_out: str
    dev_out: str
