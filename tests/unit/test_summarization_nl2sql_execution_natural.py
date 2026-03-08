from agnostic_agent.graph.summarization import build_agnostic_user_answer


def test_build_agnostic_user_answer_nl2sql_with_execution_is_natural():
    runs = [
        {
            "name": "nl2sql_sqlite",
            "args": {
                "user_request": "dame informacion sobre el credito LOC-0004",
                "entity_id": "LOC-0004",
                "execute": True,
            },
            "output": {
                "ok": True,
                "db_path": "session/contabilidad.db",
                "user_request": "dame informacion sobre el credito LOC-0004",
                "entity_id": "LOC-0004",
                "chosen_table": "estados_cuenta",
                "where_clauses": ["credito_id = 'LOC-0004'"],
                "execution": {
                    "ok": True,
                    "columns": ["credito_id", "estatus", "saldo_total"],
                    "rows": [["LOC-0004", "Desembolsado", 29440.64]],
                    "row_count": 1,
                },
            },
        }
    ]

    text = build_agnostic_user_answer("dame informacion sobre el credito LOC-0004", runs)

    assert "Encontré 1 registro" in text or "Encontr" in text
    assert "Datos clave:" not in text
