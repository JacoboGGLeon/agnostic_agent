from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from agnostic_agent.finance_skill_common import artifact, extract_credito_id
from agnostic_agent.tools.finance import finance_sources_status, get_saneamiento_rate, lookup_finance_dictionary, lookup_finance_rule
from agnostic_agent.tools.introspection import nl2sql


def _infer_intent(user_request: str) -> str:
    text = (user_request or "").lower()
    if any(tok in text for tok in ["asiento", "poliza"]):
        return "generate_accounting_entry"
    if any(tok in text for tok in ["batch", "lote", "contabilizacion", "exporta"]):
        return "export_accounting_batch"
    if any(tok in text for tok in ["cumplimiento", "normativo", "normativa", "valida"]):
        return "validate_accounting_compliance"
    return "assign_accounting_account"


@dataclass
class GobiernoCuentasContablesSkill:
    name: str = "gobierno_cuentas_contables"
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
        query_request = user_request if not credito_id else f"{user_request} {credito_id}"
        query_out = nl2sql.invoke(
            {
                "user_request": query_request,
                "db_path": "contabilidad.db",
                "row_limit": int(request.get("row_limit") or 50),
                "execute": bool(request.get("execute", True)),
                "entity_id": credito_id,
            }
        )
        query_payload = query_out if isinstance(query_out, dict) else {"raw": query_out}
        artifacts: List[Dict[str, Any]] = [artifact("query_result", query_payload)]

        if intent == "assign_accounting_account":
            dictionary = lookup_finance_dictionary.invoke({"term": str(request.get("cuenta_contable") or "cuenta_contable")})
            dictionary_payload = dictionary if isinstance(dictionary, dict) else {"raw": dictionary}
            artifacts.append(artifact("semantic_evidence", dictionary_payload))
            result = {"assignment_basis": "contract_attributes", "query_result": query_payload, "dictionary_evidence": dictionary_payload}
        elif intent == "validate_accounting_compliance":
            rules = lookup_finance_rule.invoke({"query": user_request})
            sources = finance_sources_status.invoke({})
            rules_payload = rules if isinstance(rules, dict) else {"raw": rules}
            sources_payload = sources if isinstance(sources, dict) else {"raw": sources}
            artifacts.append(artifact("semantic_evidence", rules_payload))
            artifacts.append(artifact("source_status", sources_payload))
            result = {"compliance_check": query_payload, "rule_evidence": rules_payload, "sources": sources_payload}
        elif intent == "generate_accounting_entry":
            rate = get_saneamiento_rate.invoke({"estatus": str(request.get("estatus") or "")}) if request.get("estatus") else {"found": False}
            rule = lookup_finance_rule.invoke({"query": user_request, "estatus": str(request.get("estatus") or "")})
            rate_payload = rate if isinstance(rate, dict) else {"raw": rate}
            rule_payload = rule if isinstance(rule, dict) else {"raw": rule}
            artifacts.append(artifact("semantic_evidence", rate_payload))
            artifacts.append(artifact("semantic_evidence", rule_payload))
            result = {"entry_type": "accounting_entry_proposal", "query_result": query_payload, "rate_evidence": rate_payload, "rule_evidence": rule_payload}
        else:
            sources = finance_sources_status.invoke({})
            sources_payload = sources if isinstance(sources, dict) else {"raw": sources}
            artifacts.append(artifact("source_status", sources_payload))
            result = {"batch_mode": True, "query_result": query_payload, "sources": sources_payload}

        return {
            "status": "success",
            "outputs": {"skill": self.name, "version": self.version, "world": self.name, "intent": intent, "credito_id": credito_id or None, "result": result},
            "artifacts": artifacts,
            "errors": [],
            "metrics": {},
            "children": [],
        }


def build() -> GobiernoCuentasContablesSkill:
    return GobiernoCuentasContablesSkill()
