from __future__ import annotations

"""
Contratos de I/O de alto nivel del Agnostic Deep Agent 2026.

Este módulo define:
- AgentInput:  payload de entrada para el agente.
- ToolRun:     traza de ejecución de herramientas (para vistas deep/dev).
- AgentSummary:resumen por fases del pipeline interno.
- AgentView:   una vista del resultado para un "rol" (user / deep / dev).
- AgentOutput: objeto de salida unificado con las tres vistas.

NOTAS:
- Los contratos internos más ricos (AnalyzerIntent, PlannerPlan, etc.)
  vivirán en `schemas.py`. Aquí sólo se define I/O de "frontera".
- El campo canónico es `user_prompt`, pero se acepta también `user_text`
  como alias para compatibilidad.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────

class AgentInput(BaseModel):
    """
    Input "oficial" del agente.

    - user_prompt: texto natural del usuario (nombre canónico).
    - session_id: id opcional de sesión (por si quieres manejar múltiples).
    - kb_names: lista de KBs a considerar (FAOSTAT_WORLD, etc.).
    - metadata: extras arbitrarios (canal, idioma detectado, flags, etc.).
    - data_payload: payload estructurado para este turno
        (por ejemplo, fila de atributos, texto OCR, tablas, etc.).

    Compatibilidad:
    - También acepta el alias `user_text` al instanciar el modelo.
    """

    # Permite poblar usando el nombre de campo o el alias
    model_config = ConfigDict(populate_by_name=True)

    user_prompt: str = Field(
        ...,
        description="Texto de entrada del usuario.",
        alias="user_text",  # compat: antes usábamos `user_text`
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Identificador opcional de sesión.",
    )
    kb_names: List[str] = Field(
        default_factory=list,
        description="Nombres de Knowledge Bases relevantes.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadatos arbitrarios asociados a la petición.",
    )
    data_payload: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Payload estructurado asociado al turno. "
            "Ejemplos: una fila de atributos, texto OCR, "
            "tablas de entrada, IDs de contrato, etc."
        ),
    )


# ─────────────────────────────────────────────
# TRAZAS DE TOOLS + RESUMEN
# ─────────────────────────────────────────────

class ToolRun(BaseModel):
    """
    Ejecución de una herramienta.

    - id: tool_call_id interno o identificador de step.
    - name: nombre de la tool.
    - args: argumentos con los que se llamó (ya resueltos).
    - output: salida cruda (sin postprocesar).
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

    # ─────────────────────────────────────
    # Helper para construir desde AgentState
    # ─────────────────────────────────────
    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "AgentOutput":
        """
        Crea un AgentOutput a partir del estado interno del grafo.

        Convenciones esperadas (pero tolerantes):
        - state["user_out"], state["deep_out"], state["dev_out"]:
            respuestas finales (str) por rol.
          Alternativamente:
            state["user_answer"], state["deep_answer"], state["dev_answer"].
        - state["tool_runs"]:
            lista de ToolRun, dicts o estructuras similares.
        - summaries opcionales:
            puedes mapear campos del state a AgentSummary si lo deseas
            vía `state["pipeline_summary"]`.
        """
        # Tool runs normalizados
        tool_runs_norm: List[ToolRun] = []
        raw_tool_runs = state.get("tool_runs", []) or []

        for tr in raw_tool_runs:
            if isinstance(tr, ToolRun):
                tool_runs_norm.append(tr)
            elif isinstance(tr, BaseModel):
                tool_runs_norm.append(ToolRun.model_validate(tr.model_dump()))
            elif isinstance(tr, dict):
                # Puede venir con claves extra, las ignoramos
                tool_runs_norm.append(ToolRun.model_validate(tr))
            else:
                # Si es algo raro, lo metemos como output genérico
                tool_runs_norm.append(
                    ToolRun(
                        id=str(len(tool_runs_norm)),
                        name="unknown",
                        args={},
                        output=tr,
                    )
                )

        # Summary opcional, si el grafo lo llena
        summary: Optional[AgentSummary] = None
        summary_data = state.get("pipeline_summary")
        if isinstance(summary_data, AgentSummary):
            summary = summary_data
        elif isinstance(summary_data, BaseModel):
            summary = AgentSummary.model_validate(summary_data.model_dump())
        elif isinstance(summary_data, dict):
            summary = AgentSummary.model_validate(summary_data)

        # Vistas
        user_view = AgentView(
            final_answer=state.get("user_out", "") or state.get("user_answer", ""),
            summary=summary,
            tool_runs=[],  # por defecto no cargamos tool_runs en la vista de usuario
            raw_state={},
        )

        deep_view = AgentView(
            final_answer=state.get("deep_out", "") or state.get("deep_answer", ""),
            summary=summary,
            tool_runs=tool_runs_norm,
            raw_state={},  # opcionalmente podrías incluir fragmentos de estado aquí
        )

        dev_view = AgentView(
            final_answer=state.get("dev_out", "") or state.get("dev_answer", ""),
            summary=summary,
            tool_runs=tool_runs_norm,
            raw_state=state,  # vista de desarrollo: estado crudo completo
        )

        return cls(dev_out=dev_view, deep_out=deep_view, user_out=user_view)
