from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from agnostic_agent.finance_skill_common import artifact, extract_credito_id
from agnostic_agent.tools.finance import get_saneamiento_rate, lookup_finance_rule, reconcile_credit_accounting
from agnostic_agent.tools.introspection import nl2sql


def _infer_intent(user_request: str) -> str:
    text = (user_request or "").lower()
    if any(tok in text for tok in ["liberacion", "liberaciones", "dotacion", "dotaciones"]):
        return "identify_liberaciones_dotaciones"
    if any(tok in text for tok in ["mensual", "mes", "resultado", "resultados"]):
        return "calculate_monthly_saneamiento_cost"
    if any(tok in text for tok in ["trazabilidad", "traza", "rastreo", "por contrato"]):
        return "contract_traceability"
    return "explain_saneamiento_cost"


@dataclass
class CostoSaneamientoContratoSkill:
    name: str = "costo_saneamiento_contrato"
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

        if intent == "calculate_monthly_saneamiento_cost":
            result = nl2sql.invoke(
                {
                    "user_request": user_request,
                    "db_path": "contabilidad.db",
                    "row_limit": int(request.get("row_limit") or 50),
                    "execute": bool(request.get("execute", True)),
                }
            )
            payload = result if isinstance(result, dict) else {"raw": result}
            artifacts.append(artifact("query_result", payload))
            return {
                "status": "success",
                "outputs": {"skill": self.name, "version": self.version, "world": self.name, "intent": intent, "result": payload},
                "artifacts": artifacts,
                "errors": [],
                "metrics": {},
                "children": [],
            }

        if not credito_id:
            return {
                "status": "error",
                "outputs": {"skill": self.name, "version": self.version, "world": self.name, "intent": intent},
                "artifacts": [],
                "errors": [{"code": "MISSING_CREDITO_ID", "message": "credito_id is required for contract-level costing"}],
                "metrics": {},
                "children": [],
            }

        reconciliation = reconcile_credit_accounting.invoke({"credito_id": credito_id, "balance": str(request.get("balance") or "")})
        reconciliation_payload = reconciliation if isinstance(reconciliation, dict) else {"raw": reconciliation}
        artifacts.append(artifact("query_result", reconciliation_payload))

        estatus = str(reconciliation_payload.get("estatus") or request.get("estatus") or "").strip()
        if estatus:
            rate = get_saneamiento_rate.invoke({"estatus": estatus})
            rule = lookup_finance_rule.invoke({"query": user_request or estatus, "estatus": estatus})
            artifacts.append(artifact("semantic_evidence", rate if isinstance(rate, dict) else {"raw": rate}))
            artifacts.append(artifact("semantic_evidence", rule if isinstance(rule, dict) else {"raw": rule}))

        return {
            "status": "success",
            "outputs": {
                "skill": self.name,
                "version": self.version,
                "world": self.name,
                "intent": intent,
                "credito_id": credito_id,
                "result": reconciliation_payload,
            },
            "artifacts": artifacts,
            "errors": [],
            "metrics": {},
            "children": [],
        }


def build() -> CostoSaneamientoContratoSkill:
    return CostoSaneamientoContratoSkill()
