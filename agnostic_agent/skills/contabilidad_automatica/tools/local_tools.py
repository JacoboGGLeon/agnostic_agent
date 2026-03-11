"""Local wrapper declarations for contabilidad_automatica."""


def tool_contracts():
    return {
        "skill": "contabilidad_automatica",
        "tools": [
            "reconcile_credit_accounting",
            "query_transactions_db",
            "query_accounting_db",
            "get_saneamiento_rate",
            "finance_sources_status",
            "nl2sql",
        ],
    }
