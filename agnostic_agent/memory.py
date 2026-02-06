from __future__ import annotations

"""
Gestión de MEMORIA para el Agnostic Deep Agent 2026.

Maneja:
1. Memoria de sesión (short-term): lista de mensajes recientes (managed by LangGraph state).
2. Memoria persistente / resumen (long-term):
   - Se guarda en un dict global en RAM (simulando DB).
   - Estructura:
     {
       "session_id": {
         "summary": "Resumen de lo hablado...",
         "facts": ["Usuario se llama Juan", "Interesado en hipotecas"],
         "last_turn": "..."
       }
     }
"""

from typing import Any, Dict, List, Optional
import datetime

# "Base de datos" en memoria para el demo
# En producción, esto sería Redis / Postgres / Mongo
_MEMORY_DB: Dict[str, Any] = {}


def get_session_memory(session_id: str) -> Dict[str, Any]:
    """Recupera el estado de memoria para una sesión."""
    return _MEMORY_DB.get(session_id, {})


def update_session_memory(session_id: str, new_data: Dict[str, Any]) -> None:
    """Actualiza (merge) la memoria de una sesión."""
    current = _MEMORY_DB.get(session_id, {})
    current.update(new_data)
    _MEMORY_DB[session_id] = current


def read_memory(session_id: str) -> Dict[str, Any]:
    """
    Lee la memoria disponible para inyectarla en el contexto del agente.
    Devuelve un dict con 'summary', 'facts', etc.
    """
    return get_session_memory(session_id)


def write_memory(
    session_id: str,
    user_prompt: str,
    user_out: str,
    user_id: Optional[str] = None,
    memory_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Escribe/Actualiza la memoria al final de un turno.

    Aquí se podría llamar a un 'Memory LLM' para condensar el historial
    o extraer hechos nuevos. Para la demo/MVP, hacemos un append simple
    o un update básico.
    """
    # 1. Recuperar memoria actual
    mem = get_session_memory(session_id)
    
    # 2. Actualizar metadatos básicos
    mem["last_interaction"] = datetime.datetime.now().isoformat()
    if user_id:
        mem["user_id"] = user_id

    # 3. Lógica simple de "Facts" (placeholder)
    #    En una v2, aquí invocamos a un LLM "MemorySummarizer"
    #    que tome (mem['summary'], user_prompt, user_out) -> new_summary
    
    # Por ahora, solo guardamos el último turno completo en 'history_buffer' (opcional)
    # o actualizamos un contador, etc.
    turns = mem.get("turns_count", 0)
    mem["turns_count"] = turns + 1
    
    # Ejemplo: si el usuario dice "me llamo X", guardar en facts (mock)
    # if "me llamo" in user_prompt.lower():
    #     ...
    
    update_session_memory(session_id, mem)


def clear_memory(session_id: str) -> None:
    """Borra la memoria de la sesión."""
    if session_id in _MEMORY_DB:
        del _MEMORY_DB[session_id]
