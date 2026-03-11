from agnostic_agent.memory import clear_memory, read_memory, write_memory


def test_write_memory_builds_structured_working_memory_snapshot():
    session_id = "mem-structured"
    clear_memory(session_id)

    out_state = {
        "subquery_intents": [["query_financial_data"], ["query_financial_data"]],
        "entities_by_subquery": [
            {"credito_id": "LOC-0004", "db_files": ["contabilidad.db"]},
            {"credito_id": "LOC-0005", "db_files": ["transacciones.db"]},
        ],
        "tool_runs": [
            {
                "id": "call1",
                "name": "nl2sql",
                "args": {"db_path": "contabilidad.db"},
                "output": {"db_path": "session/contabilidad.db"},
            }
        ],
    }

    write_memory(
        session_id=session_id,
        user_prompt="dame información sobre estos créditos",
        user_out="respuesta",
        out_state=out_state,
    )

    mem = read_memory(session_id)
    working = mem["working_memory"]

    assert working["active_entities_by_type"]["credito_id"] == ["LOC-0004", "LOC-0005"]
    assert working["last_listed_entities_by_type"]["credito_id"] == ["LOC-0004", "LOC-0005"]
    assert working["last_operation"] == "data_lookup"
    assert "contabilidad.db" in working["recent_sources"]
    assert len(working["recent_turns"]) == 1

    clear_memory(session_id)


def test_write_memory_persists_finance_focus_for_single_reconciliation():
    session_id = "mem-finance-focus"
    clear_memory(session_id)

    out_state = {
        "subquery_intents": [["explain_reconciliation_result"]],
        "entities_by_subquery": [{"credito_id": "LOC-0004"}],
        "tool_runs": [
            {
                "id": "call1",
                "name": "reconcile_credit_accounting",
                "args": {"credito_id": "LOC-0004"},
                "output": {
                    "ok": True,
                    "credito_id": "LOC-0004",
                    "estatus": "Desembolsado",
                    "status": "CUADRADO (100% Match)",
                    "flujos": {"DESEMBOLSO": 100.0, "PAGO": 20.0, "PENALIZACION": 0.0, "DESCUENTO": 0.0},
                    "saldo": {"reportado": 80.0, "esperado": 80.0, "diferencia": 0.0},
                    "saneamiento": {"tasa": 0.01, "reportado": 0.8, "esperado": 0.8, "diferencia": 0.0},
                },
            }
        ],
    }

    write_memory(
        session_id=session_id,
        user_prompt="explicame detalladamente como llegaste a esto: LOC-0004",
        user_out="respuesta",
        out_state=out_state,
    )

    mem = read_memory(session_id)
    working = mem["working_memory"]

    assert working["last_focus_entity_by_type"]["credito_id"] == "LOC-0004"
    assert working["last_finance_artifact"]["credito_id"] == "LOC-0004"
    assert working["last_operation"] == "reconcile"
    assert working["recent_finance_results"][0]["credito_id"] == "LOC-0004"

    clear_memory(session_id)
