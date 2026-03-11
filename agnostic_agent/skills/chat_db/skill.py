from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from agnostic_agent.tools.introspection import inspect_sqlite_schema, nl2sql


def _infer_intent(user_request: str) -> str:
    text = (user_request or "").lower()
    if any(tok in text for tok in ["lote", "batch", "varios", "lista de"]):
        return "batch_query"
    if any(tok in text for tok in ["schema", "tabla", "columna", "estructura"]):
        return "explain_schema"
    if any(tok in text for tok in ["compar", " vs ", " contra "]):
        return "compare_entities"
    if any(tok in text for tok in ["cuanto", "cuánt", "count", "sum", "avg", "promedio", "max", "min"]):
        return "aggregate_data"
    return "query_data"


def _extract_entity_id(user_request: str, request: Dict[str, Any]) -> str:
    explicit = str(request.get("entity_id") or request.get("credito_id") or "").strip()
    if explicit:
        return explicit
    match = re.search(r"\bLOC-\d{3,}\b", user_request or "", flags=re.IGNORECASE)
    return match.group(0).upper() if match else ""


def _artifact(kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": kind, "payload": payload}


@dataclass
class ChatDBSkill:
    name: str = "chat_db"
    version: str = "1.0.0"

    def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        user_request = str(request.get("user_request") or request.get("query") or "").strip()
        if not user_request:
            return {
                "status": "error",
                "outputs": {"skill": self.name, "version": self.version, "world": "chat_db"},
                "artifacts": [],
                "errors": [{"code": "MISSING_USER_REQUEST", "message": "user_request is required"}],
                "metrics": {},
                "children": [],
            }

        db_path = str(request.get("db_path") or "").strip()
        row_limit = int(request.get("row_limit") or 50)
        intent = str(request.get("intent") or _infer_intent(user_request))
        entity_id = _extract_entity_id(user_request, request)
        artifacts: List[Dict[str, Any]] = []

        if intent == "explain_schema":
            schema_out = inspect_sqlite_schema.invoke({"db_path": db_path, "user_request": user_request})
            artifacts.append(_artifact("schema_result", schema_out if isinstance(schema_out, dict) else {"raw": schema_out}))
            return {
                "status": "success",
                "outputs": {
                    "skill": self.name,
                    "version": self.version,
                    "world": "chat_db",
                    "intent": intent,
                    "db_path": (schema_out or {}).get("db_path", db_path) if isinstance(schema_out, dict) else db_path,
                    "result": schema_out,
                },
                "artifacts": artifacts,
                "errors": [],
                "metrics": {"row_limit": row_limit},
                "children": [],
            }

        nl2sql_out = nl2sql.invoke(
            {
                "user_request": user_request,
                "db_path": db_path,
                "row_limit": row_limit,
                "execute": bool(request.get("execute", True)),
                "entity_id": entity_id,
            }
        )
        payload = nl2sql_out if isinstance(nl2sql_out, dict) else {"raw": nl2sql_out}
        artifacts.append(_artifact("sql_result", payload))
        return {
            "status": "success",
            "outputs": {
                "skill": self.name,
                "version": self.version,
                "world": "chat_db",
                "intent": intent,
                "entity_id": entity_id or None,
                "db_path": payload.get("db_path", db_path),
                "result": payload,
            },
            "artifacts": artifacts,
            "errors": [],
            "metrics": {"row_limit": row_limit},
            "children": [],
        }


def build() -> ChatDBSkill:
    return ChatDBSkill()
