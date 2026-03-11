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
