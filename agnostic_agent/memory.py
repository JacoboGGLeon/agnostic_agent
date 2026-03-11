from __future__ import annotations

"""
Structured session memory for the Agnostic Agent.

This module keeps a small in-process session store intended for the current CLI/Streamlit scope.
The important part is not persistence technology, but the shape of the stored data:

- short-term working memory is stored as typed collections
- recent turns are kept in bounded deques
- active entities/sources are preserved in normalized lists

The memory API intentionally remains simple:

- `read_memory(session_id)` returns a JSON-serializable snapshot
- `write_memory(...)` updates the working memory after a turn
- `clear_memory(session_id)` resets a session
"""

import datetime as _dt
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional

_MEMORY_DB: Dict[str, Dict[str, Any]] = {}

_RECENT_TURNS_MAXLEN = 20
_RECENT_ENTITIES_MAXLEN = 25
_RECENT_SOURCES_MAXLEN = 12
_FOCUS_STACK_MAXLEN = 10


def _ensure_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _ensure_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _ensure_deque(value: Any, *, maxlen: int) -> Deque[Any]:
    if isinstance(value, deque):
        return deque(value, maxlen=maxlen)
    if isinstance(value, list):
        return deque(value, maxlen=maxlen)
    return deque(maxlen=maxlen)


def _normalize_text(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _basename(path_like: str) -> str:
    text = _normalize_text(path_like)
    if not text:
        return ""
    return text.replace("\\", "/").split("/")[-1]


def _unique(values: Iterable[Any]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        normalized = _normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _deque_snapshot(value: Any) -> List[Any]:
    if isinstance(value, deque):
        return list(value)
    if isinstance(value, list):
        return list(value)
    return []


def _normalize_entity_groups(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: Dict[str, List[str]] = {}
    for key, items in value.items():
        key_text = _normalize_text(key)
        if not key_text:
            continue
        if isinstance(items, (list, tuple, deque)):
            normalized[key_text] = _unique(items)
        elif items not in (None, ""):
            normalized[key_text] = [_normalize_text(items)]
    return normalized


def _extract_entity_groups_from_state(out_state: Mapping[str, Any]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for entity_group in out_state.get("entities_by_subquery") or []:
        if not isinstance(entity_group, Mapping):
            continue
        for key, value in entity_group.items():
            if key == "db_files":
                continue
            if isinstance(value, list):
                grouped.setdefault(str(key), []).extend(_unique(value))
            elif value not in (None, ""):
                grouped.setdefault(str(key), []).append(_normalize_text(value))
    return {key: _unique(values) for key, values in grouped.items() if values}


def _extract_sources_from_state(out_state: Mapping[str, Any]) -> List[str]:
    sources: List[str] = []
    for entity_group in out_state.get("entities_by_subquery") or []:
        if not isinstance(entity_group, Mapping):
            continue
        for db_file in entity_group.get("db_files", []) if isinstance(entity_group.get("db_files"), list) else []:
            base = _basename(str(db_file))
            if base:
                sources.append(base)
    for run in out_state.get("tool_runs") or []:
        if not isinstance(run, Mapping):
            continue
        args = run.get("args") if isinstance(run.get("args"), Mapping) else {}
        db_path = _basename(args.get("db_path"))
        if db_path:
            sources.append(db_path)
        output = run.get("output")
        if isinstance(output, Mapping):
            output_db = _basename(output.get("db_path"))
            if output_db:
                sources.append(output_db)
            sources_map = output.get("sources") if isinstance(output.get("sources"), Mapping) else {}
            for value in sources_map.values():
                source_name = _basename(value)
                if source_name.endswith(".db"):
                    sources.append(source_name)
    return _unique(sources)


def _extract_operation_label(user_prompt: str, out_state: Mapping[str, Any]) -> str:
    prompt_text = (user_prompt or "").lower()
    intents = out_state.get("subquery_intents") or []
    flat_intents = {str(intent) for group in intents if isinstance(group, list) for intent in group}
    if any(intent in {"batch_reconcile", "reconcile_credit", "audit_drift"} for intent in flat_intents):
        return "reconcile"
    if any(intent in {"explain_rule", "semantic_lookup", "semantic_synthesis"} for intent in flat_intents):
        return "knowledge_lookup"
    if any(intent in {"query_financial_data", "query_data", "aggregate_data", "compare_entities"} for intent in flat_intents):
        return "data_lookup"
    if "concili" in prompt_text or "drift" in prompt_text:
        return "reconcile"
    if "movimiento" in prompt_text or "transaccion" in prompt_text or "transacción" in prompt_text:
        return "data_lookup"
    return "general_lookup"


def _ensure_working_memory(mem: MutableMapping[str, Any]) -> Dict[str, Any]:
    working = _ensure_dict(mem.get("working_memory"))
    working["recent_turns"] = _ensure_deque(working.get("recent_turns"), maxlen=_RECENT_TURNS_MAXLEN)
    working["recent_sources"] = _ensure_deque(working.get("recent_sources"), maxlen=_RECENT_SOURCES_MAXLEN)
    working["focus_stack"] = _ensure_deque(working.get("focus_stack"), maxlen=_FOCUS_STACK_MAXLEN)
    working["active_entities_by_type"] = _normalize_entity_groups(working.get("active_entities_by_type"))
    working["last_listed_entities_by_type"] = _normalize_entity_groups(working.get("last_listed_entities_by_type"))
    recent_entities_raw = _ensure_dict(working.get("recent_entities_by_type"))
    working["recent_entities_by_type"] = {
        key: _ensure_deque(values, maxlen=_RECENT_ENTITIES_MAXLEN)
        for key, values in _normalize_entity_groups(recent_entities_raw).items()
    }
    working["last_operation"] = _normalize_text(working.get("last_operation"))
    mem["working_memory"] = working
    return working


def _snapshot_memory(mem: Mapping[str, Any]) -> Dict[str, Any]:
    snapshot = dict(mem)
    working = _ensure_dict(snapshot.get("working_memory"))
    if working:
        working_snapshot = dict(working)
        working_snapshot["recent_turns"] = _deque_snapshot(working_snapshot.get("recent_turns"))
        working_snapshot["recent_sources"] = _deque_snapshot(working_snapshot.get("recent_sources"))
        working_snapshot["focus_stack"] = _deque_snapshot(working_snapshot.get("focus_stack"))
        working_snapshot["active_entities_by_type"] = _normalize_entity_groups(
            working_snapshot.get("active_entities_by_type")
        )
        working_snapshot["last_listed_entities_by_type"] = _normalize_entity_groups(
            working_snapshot.get("last_listed_entities_by_type")
        )
        recent_entities = _ensure_dict(working_snapshot.get("recent_entities_by_type"))
        working_snapshot["recent_entities_by_type"] = {
            key: _deque_snapshot(value) for key, value in recent_entities.items()
        }
        snapshot["working_memory"] = working_snapshot
    return snapshot


def _remember_entities(
    working: MutableMapping[str, Any],
    entity_groups: Dict[str, List[str]],
    *,
    mark_as_listed: bool,
) -> None:
    active = _normalize_entity_groups(working.get("active_entities_by_type"))
    listed = _normalize_entity_groups(working.get("last_listed_entities_by_type"))
    recent_raw = _ensure_dict(working.get("recent_entities_by_type"))

    for entity_type, values in entity_groups.items():
        if not values:
            continue
        active[entity_type] = list(values)
        if mark_as_listed:
            listed[entity_type] = list(values)
        recent_deque = _ensure_deque(recent_raw.get(entity_type), maxlen=_RECENT_ENTITIES_MAXLEN)
        for value in values:
            recent_deque.append(value)
        recent_raw[entity_type] = recent_deque

    working["active_entities_by_type"] = active
    working["last_listed_entities_by_type"] = listed
    working["recent_entities_by_type"] = recent_raw


def _append_focus(
    working: MutableMapping[str, Any],
    *,
    operation: str,
    entity_groups: Dict[str, List[str]],
    sources: List[str],
) -> None:
    focus_stack = _ensure_deque(working.get("focus_stack"), maxlen=_FOCUS_STACK_MAXLEN)
    focus_item = {
        "timestamp": _dt.datetime.now().isoformat(),
        "operation": operation,
        "entities": entity_groups,
        "sources": list(sources),
    }
    if entity_groups or sources:
        focus_stack.append(focus_item)
    working["focus_stack"] = focus_stack


def _build_turn_snapshot(
    *,
    user_prompt: str,
    user_out: str,
    out_state: Mapping[str, Any],
) -> Dict[str, Any]:
    entity_groups = _extract_entity_groups_from_state(out_state)
    sources = _extract_sources_from_state(out_state)
    operation = _extract_operation_label(user_prompt, out_state)
    return {
        "timestamp": _dt.datetime.now().isoformat(),
        "user_prompt": user_prompt,
        "user_out": user_out,
        "operation": operation,
        "entity_groups": entity_groups,
        "sources": sources,
    }


def get_session_memory(session_id: str) -> Dict[str, Any]:
    return _MEMORY_DB.get(session_id, {})


def update_session_memory(session_id: str, new_data: Dict[str, Any]) -> None:
    current = _ensure_dict(_MEMORY_DB.get(session_id))
    current.update(new_data)
    _MEMORY_DB[session_id] = current


def read_memory(session_id: str) -> Dict[str, Any]:
    return _snapshot_memory(get_session_memory(session_id))


def write_memory(
    session_id: str,
    user_prompt: str,
    user_out: str,
    user_id: Optional[str] = None,
    memory_cfg: Optional[Dict[str, Any]] = None,
    out_state: Optional[Dict[str, Any]] = None,
) -> None:
    mem = _ensure_dict(get_session_memory(session_id))
    mem["last_interaction"] = _dt.datetime.now().isoformat()
    mem["turns_count"] = int(mem.get("turns_count", 0) or 0) + 1
    if user_id:
        mem["user_id"] = user_id
    if memory_cfg:
        mem["memory_cfg"] = dict(memory_cfg)

    working = _ensure_working_memory(mem)
    state_snapshot = _ensure_dict(out_state)
    turn_snapshot = _build_turn_snapshot(
        user_prompt=user_prompt,
        user_out=user_out,
        out_state=state_snapshot,
    )

    recent_turns = _ensure_deque(working.get("recent_turns"), maxlen=_RECENT_TURNS_MAXLEN)
    recent_turns.append(turn_snapshot)
    working["recent_turns"] = recent_turns

    recent_sources = _ensure_deque(working.get("recent_sources"), maxlen=_RECENT_SOURCES_MAXLEN)
    for source in turn_snapshot["sources"]:
        recent_sources.append(source)
    working["recent_sources"] = recent_sources

    entity_groups = turn_snapshot["entity_groups"]
    mark_as_listed = any(len(values) > 1 for values in entity_groups.values())
    _remember_entities(working, entity_groups, mark_as_listed=mark_as_listed)
    working["last_operation"] = turn_snapshot["operation"]
    _append_focus(
        working,
        operation=turn_snapshot["operation"],
        entity_groups=entity_groups,
        sources=turn_snapshot["sources"],
    )

    update_session_memory(session_id, mem)


def clear_memory(session_id: str) -> None:
    if session_id in _MEMORY_DB:
        del _MEMORY_DB[session_id]
