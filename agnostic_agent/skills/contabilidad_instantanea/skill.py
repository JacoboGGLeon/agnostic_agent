from dataclasses import dataclass
from typing import Any, Dict, List


REQUIRED_FIELDS = ("credito_id", "estatus", "saldo_total")


@dataclass
class ContabilidadInstantaneaSkill:
    name: str = "contabilidad_instantanea"
    version: str = "1.1.0"

    def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        missing = [field for field in REQUIRED_FIELDS if request.get(field) in (None, "")]
        if missing:
            return {
                "status": "error",
                "outputs": {
                    "ok": False,
                    "error_code": "missing_required_fields",
                    "missing_fields": missing,
                },
                "artifacts": [],
                "errors": [{"code": "missing_required_fields", "fields": missing}],
                "metrics": {},
                "children": [],
            }

        credito_id = str(request["credito_id"])
        transacciones_sql = (
            "SELECT tipo, monto FROM movimientos "
            f"WHERE credito_id = '{credito_id}' ORDER BY fecha ASC"
        )
        contabilidad_sql = (
            "SELECT saldo_total, estatus, saneamiento_calculado FROM estados_cuenta "
            f"WHERE credito_id = '{credito_id}'"
        )
        planned_tool_calls: List[Dict[str, Any]] = [
            {
                "tool": "query_transactions_db",
                "args": {"query": transacciones_sql},
            },
            {
                "tool": "query_accounting_db",
                "args": {"query": contabilidad_sql},
            },
        ]

        return {
            "status": "success",
            "outputs": {
                "ok": True,
                "skill": self.name,
                "version": self.version,
                "credito_id": credito_id,
                "estado_conciliacion": "PENDIENTE_EJECUCION_TOOLS",
                "planned_tool_calls": planned_tool_calls,
                "persona": "Contador IA",
                "canonical_skill_in_notebook": "reconcile_accounts",
            },
            "artifacts": [],
            "errors": [],
            "metrics": {"planned_calls": 2},
            "children": [],
        }


def build() -> ContabilidadInstantaneaSkill:
    return ContabilidadInstantaneaSkill()
