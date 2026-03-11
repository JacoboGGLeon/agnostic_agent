from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from agnostic_agent.finance_skill_common import artifact, extract_credito_id, guess_finance_db
from agnostic_agent.tools.finance import finance_sources_status, get_saneamiento_rate, lookup_finance_dictionary, lookup_finance_rule, reconcile_credit_accounting
from agnostic_agent.tools.introspection import nl2sql


def _infer_intent(user_request: str) -> str:
    text = (user_request or "").lower()
    if any(tok in text for tok in ["deterioro", "mejora", "stage", "empeoro", "mejoro"]):
        return "explain_deterioration"
    if any(tok in text for tok in ["tendencia", "tendencias", "proyeccion", "proyecciones", "forecast"]):
        return "generate_trends_and_projections"
    if any(tok in text for tok in ["soporte", "incidencia", "correccion", "explicacion"]):
        return "support_and_fix_incidents"
    return "portfolio_breakdown"


@dataclass
class AnalisisSaneamientoStageSkill:
    name: str = "analisis_saneamiento_stage"
    version: str = "1.0.0"

    def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        user_request = str(request.get("user_request") or request.get("query") or "").strip()
        if not user_request and not request.get("credito_id"):
            return {
                "status": "error",
                "outputs": {"skill": self.name, "version": self.version, "world": self.name},
                "artifacts": [],
                "errors": [{"code": "MISSING_USER_REQUEST", "message": "user_request or credito_id is required"}],
                "metrics": {},
                "children": [],
            }

        intent = str(request.get("intent") or _infer_intent(user_request))
        credito_id = extract_credito_id(user_request, request)
        artifacts: List[Dict[str, Any]] = []

        if intent == "explain_deterioration" and credito_id:
            reconciliation = reconcile_credit_accounting.invoke({"credito_id": credito_id, "balance": str(request.get("balance") or "")})
            reconciliation_payload = reconciliation if isinstance(reconciliation, dict) else {"raw": reconciliation}
            estatus = str(reconciliation_payload.get("estatus") or request.get("estatus") or "").strip()
            rate = get_saneamiento_rate.invoke({"estatus": estatus}) if estatus else {"found": False}
            rule = lookup_finance_rule.invoke({"query": user_request or estatus, "estatus": estatus})
            rate_payload = rate if isinstance(rate, dict) else {"raw": rate}
            rule_payload = rule if isinstance(rule, dict) else {"raw": rule}
            artifacts.append(artifact("query_result", reconciliation_payload))
            artifacts.append(artifact("semantic_evidence", rate_payload))
            artifacts.append(artifact("semantic_evidence", rule_payload))
            result = {"credito_id": credito_id, "reconciliation": reconciliation_payload, "rate_evidence": rate_payload, "rule_evidence": rule_payload}
        else:
            query_out = nl2sql.invoke(
                {
                    "user_request": user_request,
                    "db_path": guess_finance_db(user_request, request),
                    "row_limit": int(request.get("row_limit") or 50),
                    "execute": bool(request.get("execute", True)),
                    "entity_id": credito_id,
                }
            )
            query_payload = query_out if isinstance(query_out, dict) else {"raw": query_out}
            artifacts.append(artifact("query_result", query_payload))
            result = query_payload
            if intent == "support_and_fix_incidents":
                dictionary = lookup_finance_dictionary.invoke({"term": str(request.get("term") or "saneamiento_calculado")})
                sources = finance_sources_status.invoke({})
                dictionary_payload = dictionary if isinstance(dictionary, dict) else {"raw": dictionary}
                sources_payload = sources if isinstance(sources, dict) else {"raw": sources}
                artifacts.append(artifact("semantic_evidence", dictionary_payload))
                artifacts.append(artifact("source_status", sources_payload))
                result = {"query_result": query_payload, "dictionary_evidence": dictionary_payload, "sources": sources_payload}

        return {
            "status": "success",
            "outputs": {"skill": self.name, "version": self.version, "world": self.name, "intent": intent, "credito_id": credito_id or None, "result": result},
            "artifacts": artifacts,
            "errors": [],
            "metrics": {},
            "children": [],
        }


def build() -> AnalisisSaneamientoStageSkill:
    return AnalisisSaneamientoStageSkill()
