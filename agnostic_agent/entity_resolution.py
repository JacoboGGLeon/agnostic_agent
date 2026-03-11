from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Tuple

from agnostic_agent.tools.finance import get_known_finance_statuses


_GENERIC_ID_TOKEN_RE = re.compile(r"\b[A-Za-z]{2,}[A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b")
_DB_PATH_RE = re.compile(r"\b([A-Za-z0-9_.-]+\.db)\b", flags=re.IGNORECASE)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = text.replace("â€“", "-").replace("â€”", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_json_entities(text: str) -> Dict[str, Any]:
    entities: Dict[str, Any] = {}
    for chunk in re.findall(r"\{[^{}]+\}", text or ""):
        try:
            parsed = json.loads(chunk)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        for key, value in parsed.items():
            if value not in (None, ""):
                entities[str(key)] = value
    return entities


def _extract_explicit_field_value(text: str, field_name: str) -> str:
    pattern = re.compile(
        rf"\b{re.escape(field_name)}\b\s*[:=]\s*([A-Za-z0-9_.\- /()]+)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def _resolve_id_like_entity(entity_name: str, text: str, existing_entities: Dict[str, Any]) -> str:
    explicit = str(existing_entities.get(entity_name) or "").strip()
    if explicit:
        return explicit
    explicit_field = _extract_explicit_field_value(text, entity_name)
    if explicit_field:
        return explicit_field
    candidates = [match.group(0).strip() for match in _GENERIC_ID_TOKEN_RE.finditer(text or "") if match.group(0).strip()]
    if entity_name == "credito_id":
        for candidate in candidates:
            if candidate.upper().startswith("LOC-"):
                return candidate.upper()
    return candidates[0] if candidates else ""


def _resolve_db_path(text: str, existing_entities: Dict[str, Any]) -> str:
    explicit = str(existing_entities.get("db_path") or "").strip()
    if explicit:
        return explicit
    match = _DB_PATH_RE.search(text or "")
    return match.group(1) if match else ""


def _resolve_estatus(text: str, existing_entities: Dict[str, Any]) -> str:
    explicit = str(existing_entities.get("estatus") or "").strip()
    if explicit:
        return explicit
    explicit_field = _extract_explicit_field_value(text, "estatus")
    if explicit_field:
        return explicit_field
    normalized_query = normalize_text(text)
    for status in get_known_finance_statuses():
        if normalize_text(status) in normalized_query:
            return status
    if "para " in normalized_query:
        suffix = normalized_query.split("para ", 1)[1].strip()
        for status in get_known_finance_statuses():
            if normalize_text(status).startswith(suffix) or suffix.startswith(normalize_text(status)):
                return status
    return ""


def resolve_required_entities(
    *,
    subquery_text: str,
    intents: List[str],
    world_contract: Dict[str, Any],
    memory_context: Dict[str, Any] | None = None,
    existing_entities: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    del memory_context  # memory resolution is handled upstream in analyzer
    requirements = world_contract.get("intent_entity_requirements") if isinstance(world_contract, dict) else {}
    existing = dict(existing_entities or {})
    resolved = _extract_json_entities(subquery_text)
    resolved.update({key: value for key, value in existing.items() if value not in (None, "", [])})

    required_fields: List[str] = []
    if isinstance(requirements, dict):
        for intent in intents or []:
            intent_cfg = requirements.get(intent)
            if not isinstance(intent_cfg, dict):
                continue
            for field_name in intent_cfg.get("required", []) if isinstance(intent_cfg.get("required"), list) else []:
                normalized = str(field_name).strip()
                if normalized.endswith("[]"):
                    normalized = normalized[:-2]
                if normalized and normalized not in required_fields:
                    required_fields.append(normalized)

    for entity_name in required_fields:
        if resolved.get(entity_name) not in (None, "", []):
            continue
        if entity_name.endswith("_id"):
            value = _resolve_id_like_entity(entity_name, subquery_text, resolved)
        elif entity_name == "estatus":
            value = _resolve_estatus(subquery_text, resolved)
        elif entity_name == "db_path":
            value = _resolve_db_path(subquery_text, resolved)
        else:
            value = _extract_explicit_field_value(subquery_text, entity_name)
        if value:
            resolved[entity_name] = value

    missing = [field_name for field_name in required_fields if resolved.get(field_name) in (None, "", [])]
    return {
        "resolved_entities": resolved,
        "required_entities": required_fields,
        "missing_entities": missing,
    }


def planner_block_reason(*, missing_entities: List[str]) -> str:
    if missing_entities:
        return "missing_required_entity"
    return ""
