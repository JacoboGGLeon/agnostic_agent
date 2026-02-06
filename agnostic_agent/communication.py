from __future__ import annotations

"""
Modelos de datos para la COMUNICACIÓN del Agnostic Deep Agent 2026.

Incluye:
- BaseAgentInput: definición (Pydantic) de lo que recibe el agente.
- AgentInput: alias/wrapper principal.
- AgentView: la "vista" de salida (raw / user / deep).
- AgentOutput: el objeto final unificado que devuelve el agente.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────

class BaseAgentInput(BaseModel):
    """
    Estructura robusta de la entrada del agente.
    Permite invocarlo con un dict simple o con un objeto tipado.
    """
    prompt: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Rutas a archivos de contexto (KBs, tablas CSV, etc.)
    # Se usan si el usuario quiere inyectar contexto "al vuelo"
    # aparte del configurado en setup.yaml.
    context_files: List[str] = Field(default_factory=list)
    
    # Datos extra arbitrarios (flags, override de config, etc.)
    extra_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Alias
    user_prompt: Optional[str] = None
    user_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    kb_names: List[str] = Field(default_factory=list)


# Alias principal para uso en firmas
AgentInput = BaseAgentInput


# ─────────────────────────────────────────────
# RUNS (Trazas de herramientas)
# ─────────────────────────────────────────────

class ToolRun(BaseModel):
    """
    Representa la ejecución de una herramienta.
    Normalizado para las vistas.
    """
    id: str
    name: str
    args: Dict[str, Any]
    output: Any


class AgentSummary(BaseModel):
    """
    Resumen estructurado del pipeline:

      ANALYZER → PLANNER → EXECUTOR → CATCHER → VALIDATOR → MEMORY → SUMMARIZER
    """
    analyzer: str = ""
    planner: str = ""
    executor: str = ""
    catcher: str = ""
    validator: str = ""   # nuevo nodo explícito
    memory: str = ""      # resumen de escritura/uso de memoria
    summarizer: str = ""
    final_answer: str = ""


# ─────────────────────────────────────────────
# VISTAS POR ROL
# ─────────────────────────────────────────────

class AgentView(BaseModel):
    """
    Vista del agente para un "rol" (dev / deep / user).

    - final_answer: texto final para ese rol.
    - summary: breakdown interno del pipeline.
    - tool_runs: ejecuciones de tools (ya normalizadas).
    - raw_state: estado crudo del grafo (según rol, puede ir vacío).
    """
    final_answer: str = ""
    summary: Optional[AgentSummary] = None
    tool_runs: List[ToolRun] = Field(default_factory=list)
    raw_state: Dict[str, Any] = Field(default_factory=dict)
    
    # Compatibilidad con código previo que usaba content/metadata
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        # Fallback simple: si viene 'content', lo movemos a final_answer si está vacío
        if "content" in data and "final_answer" not in data:
            data["final_answer"] = data["content"]
        super().__init__(**data)
        if not self.content and self.final_answer:
            self.content = self.final_answer


# ─────────────────────────────────────────────
# OUTPUT UNIFICADO
# ─────────────────────────────────────────────

class AgentOutput(BaseModel):
    """
    Output "oficial" del agente (UN SOLO OBJETO):

      {
        "dev_out":  AgentView(...),  # traza completa (pipeline + raw_state)
        "deep_out": AgentView(...),  # resumen por sección
        "user_out": AgentView(...),  # respuesta 1:1 para usuario final
      }
    """
    dev_out: AgentView
    deep_out: AgentView
    user_out: AgentView

    def to_dict(self) -> Dict[str, Any]:
        """
        Devuelve un dict puro {dev_out, deep_out, user_out}, listo para JSON / logging.
        """
        return self.model_dump()
