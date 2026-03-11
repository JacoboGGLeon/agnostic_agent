from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from agnostic_agent.finance_skill_common import artifact, extract_credito_id, guess_finance_db
from agnostic_agent.tools.finance import finance_sources_status, reconcile_credit_accounting
from agnostic_agent.tools.introspection import nl2sql


def _infer_intent(user_request: str) -> str:
    text = (user_request or "").lower()
    if any(tok in text for tok in ["remediacion", "remediar", "escalar", "escalamiento"]):
        return "propose_remediation"
    if any(tok in text for tok in ["incidencia", "incidencias", "recurrente", "recurrentes", "bitacora"]):
        return "analyze_recurring_incidents"
    if any(tok in text for tok in ["variacion", "variaciones", "atipica", "atipicas", "alerta", "alertas"]):
        return "detect_significant_variations"
    if any(tok in text for tok in ["inventario", "vs contable"]):
        return "inventory_vs_accounting"
    return "logical_reconciliation"


@dataclass
class ConciliacionesAlertasSkill:
    name: str = "conciliaciones_alertas"
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

        if intent in {"logical_reconciliation", "propose_remediation"} and credito_id:
            reconciliation = reconcile_credit_accounting.invoke({"credito_id": credito_id, "balance": str(request.get("balance") or "")})
            reconciliation_payload = reconciliation if isinstance(reconciliation, dict) else {"raw": reconciliation}
            artifacts.append(artifact("query_result", reconciliation_payload))
            if intent == "propose_remediation":
                sources = finance_sources_status.invoke({})
                sources_payload = sources if isinstance(sources, dict) else {"raw": sources}
                artifacts.append(artifact("source_status", sources_payload))
                result = {"credito_id": credito_id, "reconciliation": reconciliation_payload, "sources": sources_payload}
            else:
                result = reconciliation_payload
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
            if intent == "analyze_recurring_incidents":
                sources = finance_sources_status.invoke({})
                sources_payload = sources if isinstance(sources, dict) else {"raw": sources}
                artifacts.append(artifact("source_status", sources_payload))
                result = {"query_result": query_payload, "sources": sources_payload}
            else:
                result = query_payload

        return {
            "status": "success",
            "outputs": {"skill": self.name, "version": self.version, "world": self.name, "intent": intent, "credito_id": credito_id or None, "result": result},
            "artifacts": artifacts,
            "errors": [],
            "metrics": {},
            "children": [],
        }


def build() -> ConciliacionesAlertasSkill:
    return ConciliacionesAlertasSkill()
