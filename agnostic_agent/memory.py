from __future__ import annotations

"""
Memoria del Agnostic Deep Agent 2026.

Diseño multi-nivel (versión mínima):

- Session memory:
    Historial reciente de la conversación (texto plano).
- Short-term memory:
    Snippets/resúmenes recientes cada N turnos.
- Long-term memory:
    Placeholder para recuerdos persistentes (vector store, SQL, etc.).

Este módulo NO depende de LangGraph directamente; sólo expone funciones
para leer/escribir memoria usando estructuras sencillas, en términos de:

- session_id: identificador de la conversación.
- MemoryContext: modelo Pydantic definido en schemas.py.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from .schemas import MemoryContext


# ─────────────────────────────────────────────
# Almacenamiento in-memory (versión Colab / prototipo)
# ─────────────────────────────────────────────

# Historial de mensajes por sesión (texto plano)
#   _SESSION_STORE[session_id] = ["USER: ...", "AGENT: ...", ...]
_SESSION_STORE: Dict[str, List[str]] = {}

# Snippets/resúmenes de corto plazo por sesión
#   _SHORT_TERM_STORE[session_id] = ["resumen 1", "resumen 2", ...]
_SHORT_TERM_STORE: Dict[str, List[str]] = {}

# Long-term docs por usuario (placeholder; lo normal será un vector store)
#   _LONG_TERM_STORE[user_id] = [
#       {"text": "...", "created_at": "...", "meta": {...}},
#       ...
#   ]
_LONG_TERM_STORE: Dict[str, List[Dict[str, Any]]] = {}


# ─────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _truncate_list(items: List[Any], max_len: int) -> List[Any]:
    if max_len <= 0:
        return []
    if len(items) <= max_len:
        return items
    return items[-max_len:]


def _build_short_term_snippet(user_text: str, agent_text: str, turn_idx: int) -> str:
    """
    Snippet simple de corto plazo: pensado como placeholder.

    Puedes reemplazar esto más adelante por un LLM summarizer dedicado.
    """
    u = user_text.strip().replace("\n", " ")
    a = agent_text.strip().replace("\n", " ")
    if len(u) > 120:
        u = u[:117] + "..."
    if len(a) > 160:
        a = a[:157] + "..."
    return f"[turn {turn_idx}] USER: {u} | AGENT: {a}"


# ─────────────────────────────────────────────
# Lectura de memoria
# ─────────────────────────────────────────────

def read_memory(
    session_id: str,
    *,
    max_session_messages: int = 50,
    max_short_snippets: int = 20,
) -> MemoryContext:
    """
    Construye un MemoryContext para una sesión dada.

    - max_session_messages: cuántas líneas de historial traer (USER/AGENT mezcladas).
    - max_short_snippets:  cuántos snippets de short-term traer.
    """
    session_msgs = _SESSION_STORE.get(session_id, [])
    short_snips = _SHORT_TERM_STORE.get(session_id, [])

    session_history = _truncate_list(session_msgs, max_session_messages)
    short_term_snippets = _truncate_list(short_snips, max_short_snippets)

    # Long-term: de momento no filtramos por relevancia aquí; eso se hará
    # en otro módulo (vector store, RAG, etc.). Por ahora sólo devolvemos textos.
    long_term_texts: List[str] = []
    # Nota: la clave para LONG_TERM_STORE es user_id, no session_id;
    # aquí no lo conocemos, así que devolvemos lista vacía.
    # Más adelante se puede ampliar la firma para incluir user_id.

    return MemoryContext(
        session_history=session_history,
        short_term_snippets=short_term_snippets,
        long_term_snippets=long_term_texts,
    )


# ─────────────────────────────────────────────
# Escritura / actualización de memoria
# ─────────────────────────────────────────────

def write_memory(
    session_id: str,
    *,
    user_prompt: str,
    user_out: str,
    user_id: Optional[str] = None,
    memory_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Actualiza las memorias tras un turno completado.

    Parámetros:
    - session_id: id de sesión.
    - user_prompt: último mensaje del usuario.
    - user_out: respuesta final del agente para ese turno.
    - user_id: id lógico del usuario (para long-term).
    - memory_cfg: configuración opcional con la forma:

        {
          "session": {
            "enabled": true,
            "max_messages": 50,
          },
          "short_term": {
            "enabled": true,
            "summarize_every_n_turns": 5,
            "window_turns": 20,
          },
          "long_term": {
            "enabled": false,
            "max_docs": 200,
          }
        }

      Si algún bloque o campo no está presente, se usan valores por defecto.
    """
    cfg = memory_cfg or {}

    sess_cfg = cfg.get("session", {}) or {}
    st_cfg = cfg.get("short_term", {}) or {}
    lt_cfg = cfg.get("long_term", {}) or {}

    session_enabled: bool = bool(sess_cfg.get("enabled", True))
    max_messages: int = int(sess_cfg.get("max_messages", 50))

    short_enabled: bool = bool(st_cfg.get("enabled", True))
    summarize_every_n_turns: int = int(st_cfg.get("summarize_every_n_turns", 5))
    window_turns: int = int(st_cfg.get("window_turns", 20))

    long_enabled: bool = bool(lt_cfg.get("enabled", False))
    long_max_docs: int = int(lt_cfg.get("max_docs", 200))

    # -------------------------
    # 1) Session memory
    # -------------------------
    if session_enabled:
        msgs = _SESSION_STORE.setdefault(session_id, [])
        msgs.append(f"USER: {user_prompt}")
        msgs.append(f"AGENT: {user_out}")
        # Recortamos a las últimas `max_messages` líneas
        _SESSION_STORE[session_id] = _truncate_list(msgs, max_messages)

    # Número de turnos aproximado (USER/AGENT cuentan como dos mensajes)
    num_turns = len(_SESSION_STORE.get(session_id, [])) // 2 if session_enabled else 0

    # -------------------------
    # 2) Short-term memory
    # -------------------------
    if short_enabled and num_turns > 0:
        # Cada N turnos, generamos un nuevo snippet
        if summarize_every_n_turns > 0 and num_turns % summarize_every_n_turns == 0:
            snippet = _build_short_term_snippet(user_prompt, user_out, num_turns)
            st_list = _SHORT_TERM_STORE.setdefault(session_id, [])
            st_list.append(snippet)
            # Recortamos por ventana de turnos (como proxy)
            _SHORT_TERM_STORE[session_id] = _truncate_list(st_list, window_turns)

    # -------------------------
    # 3) Long-term memory (placeholder)
    # -------------------------
    if long_enabled and user_id:
        docs = _LONG_TERM_STORE.setdefault(user_id, [])
        # Heurística mínima: guardar sólo turnos cuyo user_out tenga cierta longitud
        if len(user_out.strip()) >= 40:
            docs.append(
                {
                    "text": f"USER: {user_prompt.strip()}\nAGENT: {user_out.strip()}",
                    "created_at": _now_iso(),
                    "meta": {
                        "session_id": session_id,
                        "turn_index": num_turns,
                    },
                }
            )
            _LONG_TERM_STORE[user_id] = _truncate_list(docs, long_max_docs)


# ─────────────────────────────────────────────
# Utilidades varias
# ─────────────────────────────────────────────

def clear_session_memory(session_id: str) -> None:
    """
    Elimina TODA la memoria asociada a una sesión concreta
    (session history + short-term snippets).
    """
    _SESSION_STORE.pop(session_id, None)
    _SHORT_TERM_STORE.pop(session_id, None)


def clear_long_term_memory(user_id: str) -> None:
    """
    Elimina la memoria de largo plazo de un usuario concreto.
    """
    _LONG_TERM_STORE.pop(user_id, None)


def debug_dump_memory() -> Dict[str, Any]:
    """
    Devuelve una vista simple de todas las memorias en memoria (para depuración).
    NO usar en producción si el contenido es sensible.
    """
    return {
        "session_store": dict(_SESSION_STORE),
        "short_term_store": dict(_SHORT_TERM_STORE),
        "long_term_store": dict(_LONG_TERM_STORE),
    }
