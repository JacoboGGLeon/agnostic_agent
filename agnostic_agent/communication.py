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


# Alias principal para uso en firmas
AgentInput = BaseAgentInput


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────

class AgentView(BaseModel):
    """
    Representa una 'vista' de la respuesta del agente.
    Puede ser 'raw' (dev), 'deep' (técnico) o 'user' (final).
    """
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
