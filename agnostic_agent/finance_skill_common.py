from __future__ import annotations

import re
from typing import Any, Dict

from agnostic_agent.entity_resolution import resolve_required_entities


def artifact(kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": kind, "payload": payload}


def extract_credito_id(user_request: str, request: Dict[str, Any]) -> str:
    explicit = str(request.get("credito_id") or request.get("entity_id") or "").strip()
    if explicit:
        return explicit
    match = re.search(r"\bLOC-\d{3,}\b", user_request or "", flags=re.IGNORECASE)
    return match.group(0).upper() if match else ""


def extract_estatus(user_request: str, request: Dict[str, Any]) -> str:
    explicit = str(request.get("estatus") or "").strip()
    contract = {
        "intent_entity_requirements": {"generic": {"required": ["estatus"]}},
        "entities": ["estatus"],
    }
    resolved = resolve_required_entities(
        subquery_text=user_request,
        intents=["generic"],
        world_contract=contract,
        existing_entities={"estatus": explicit},
    )
    return str((resolved.get("resolved_entities") or {}).get("estatus") or "").strip()


def guess_finance_db(user_request: str, request: Dict[str, Any], default_db: str = "contabilidad.db") -> str:
    explicit = str(request.get("db_path") or "").strip()
    if explicit:
        return explicit
    text = (user_request or "").lower()
    if any(
        tok in text
        for tok in [
            "transaccion",
            "movimiento",
            "movimientos",
            "pago",
            "pagos",
            "desembolso",
            "desembolsos",
            "penalizacion",
            "penalizaciones",
            "descuento",
            "descuentos",
            "fecha",
            "batch",
        ]
    ):
        return "transacciones.db"
    return default_db
