from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from agnostic_agent.entity_resolution import resolve_required_entities
from agnostic_agent.tools.finance import (
    finance_sources_status,
    get_saneamiento_rate,
    lookup_finance_dictionary,
    lookup_finance_rule,
    reconcile_credit_accounting,
)
from agnostic_agent.tools.introspection import nl2sql


def _infer_intent(user_request: str) -> str:
    text = (user_request or "").lower()
    if any(tok in text for tok in ["flujo", "flujos"]) and (
        "concili" in text or "loc-" in text or "resultado" in text or "detalle" in text
    ):
        return "explain_reconciliation_flows"
    if any(tok in text for tok in ["como llegaste", "cómo llegaste", "detalle", "detall", "explicame", "explícame"]) and (
        "concili" in text or "loc-" in text or "resultado" in text or "cuadrado" in text
    ):
        return "explain_reconciliation_result"
    if any(tok in text for tok in ["drift", "descuadre", "concili", "cuadra"]):
        return "reconcile_credit"
    if any(tok in text for tok in ["regla", "tasa", "saneamiento", "reserva esperada"]):
        return "explain_rule"
    if any(tok in text for tok in ["lote", "batch", "varios creditos", "varios créditos"]):
        return "batch_reconcile"
    return "query_financial_data"


def _extract_credito_id(user_request: str, request: Dict[str, Any]) -> str:
    explicit = str(request.get("credito_id") or request.get("entity_id") or "").strip()
    if explicit:
        return explicit
    match = re.search(r"\bLOC-\d{3,}\b", user_request or "", flags=re.IGNORECASE)
    return match.group(0).upper() if match else ""


def _extract_estatus(user_request: str, request: Dict[str, Any]) -> str:
    explicit = str(request.get("estatus") or "").strip()
    contract = {
        "intent_entity_requirements": {"explain_rule": {"required": ["estatus"]}},
        "entities": ["estatus"],
    }
    resolved = resolve_required_entities(
        subquery_text=user_request,
        intents=["explain_rule"],
        world_contract=contract,
        existing_entities={"estatus": explicit},
    )
    return str((resolved.get("resolved_entities") or {}).get("estatus") or "").strip()


def _guess_finance_db(user_request: str, request: Dict[str, Any]) -> str:
    explicit = str(request.get("db_path") or "").strip()
    if explicit:
        return explicit
    text = (user_request or "").lower()
    if any(tok in text for tok in ["transaccion", "transacción", "movimiento", "pago", "desembolso", "penalizacion", "descuento", "fecha"]):
        return "transacciones.db"
    return "contabilidad.db"


def _artifact(kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": kind, "payload": payload}


@dataclass
class ContabilidadAutomaticaSkill:
    name: str = "contabilidad_automatica"
    version: str = "1.0.0"

    def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        user_request = str(request.get("user_request") or request.get("query") or "").strip()
        if not user_request and not request.get("credito_id"):
            return {
                "status": "error",
                "outputs": {"skill": self.name, "version": self.version, "world": "contabilidad_automatica"},
                "artifacts": [],
                "errors": [{"code": "MISSING_USER_REQUEST", "message": "user_request or credito_id is required"}],
                "metrics": {},
                "children": [],
            }

        intent = str(request.get("intent") or _infer_intent(user_request))
        credito_id = _extract_credito_id(user_request, request)
        estatus = _extract_estatus(user_request, request)
        artifacts: List[Dict[str, Any]] = []

        if intent in {"reconcile_credit", "audit_drift", "batch_reconcile", "explain_reconciliation_result", "explain_reconciliation_flows"}:
            if not credito_id:
                return {
                    "status": "error",
                    "outputs": {"skill": self.name, "version": self.version, "world": "contabilidad_automatica", "intent": intent},
                    "artifacts": [],
                    "errors": [{"code": "MISSING_CREDITO_ID", "message": "credito_id is required for reconciliation"}],
                    "metrics": {},
                    "children": [],
                }
            rec = reconcile_credit_accounting.invoke({"credito_id": credito_id, "balance": str(request.get("balance") or "")})
            payload = rec if isinstance(rec, dict) else {"raw": rec}
            artifacts.append(_artifact("query_result", payload))
            return {
                "status": "success",
                "outputs": {
                    "skill": self.name,
                    "version": self.version,
                    "world": "contabilidad_automatica",
                    "intent": intent,
                    "credito_id": credito_id,
                    "result": payload,
                },
                "artifacts": artifacts,
                "errors": [],
                "metrics": {},
                "children": [],
            }

        if intent == "explain_rule":
            if not estatus:
                dictionary_lookup = lookup_finance_dictionary.invoke({"term": user_request})
                dictionary_payload = dictionary_lookup if isinstance(dictionary_lookup, dict) else {"raw": dictionary_lookup}
                artifacts.append(_artifact("semantic_evidence", dictionary_payload))
                return {
                    "status": "success",
                    "outputs": {
                        "skill": self.name,
                        "version": self.version,
                        "world": "contabilidad_automatica",
                        "intent": intent,
                        "dictionary_evidence": dictionary_payload,
                    },
                    "artifacts": artifacts,
                    "errors": [],
                    "metrics": {},
                    "children": [],
                }
            rule = get_saneamiento_rate.invoke({"estatus": estatus})
            rule_lookup = lookup_finance_rule.invoke({"query": user_request or estatus, "estatus": estatus})
            dictionary_lookup = lookup_finance_dictionary.invoke({"term": "estatus"})
            sources = finance_sources_status.invoke({})
            rule_payload = rule if isinstance(rule, dict) else {"raw": rule}
            lookup_payload = rule_lookup if isinstance(rule_lookup, dict) else {"raw": rule_lookup}
            dictionary_payload = dictionary_lookup if isinstance(dictionary_lookup, dict) else {"raw": dictionary_lookup}
            sources_payload = sources if isinstance(sources, dict) else {"raw": sources}
            artifacts.append(_artifact("query_result", rule_payload))
            artifacts.append(_artifact("semantic_evidence", lookup_payload))
            artifacts.append(_artifact("semantic_evidence", dictionary_payload))
            artifacts.append(_artifact("source_status", sources_payload))
            return {
                "status": "success",
                "outputs": {
                    "skill": self.name,
                    "version": self.version,
                    "world": "contabilidad_automatica",
                    "intent": intent,
                    "estatus": estatus,
                    "result": rule_payload,
                    "rule_evidence": lookup_payload,
                    "dictionary_evidence": dictionary_payload,
                    "sources": sources_payload,
                },
                "artifacts": artifacts,
                "errors": [],
                "metrics": {},
                "children": [],
            }

        db_path = _guess_finance_db(user_request, request)
        nl_out = nl2sql.invoke(
            {
                "user_request": user_request,
                "db_path": db_path,
                "row_limit": int(request.get("row_limit") or 50),
                "execute": bool(request.get("execute", True)),
                "entity_id": credito_id,
            }
        )
        payload = nl_out if isinstance(nl_out, dict) else {"raw": nl_out}
        artifacts.append(_artifact("query_result", payload))
        return {
            "status": "success",
            "outputs": {
                "skill": self.name,
                "version": self.version,
                "world": "contabilidad_automatica",
                "intent": "query_financial_data",
                "credito_id": credito_id or None,
                "db_path": payload.get("db_path", db_path),
                "result": payload,
            },
            "artifacts": artifacts,
            "errors": [],
            "metrics": {},
            "children": [],
        }


def build() -> ContabilidadAutomaticaSkill:
    return ContabilidadAutomaticaSkill()
