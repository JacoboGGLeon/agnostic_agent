from agnostic_agent.graph.summarization import build_agnostic_user_answer


def test_build_agnostic_user_answer_mentions_missing_credit_filter_for_nl2sql():
    runs = [
        {
            "name": "nl2sql_sqlite",
            "args": {
                "user_request": "dame informacion sobre el credito LOC-0004",
                "entity_id": "LOC-0004",
            },
            "output": {
                "ok": True,
                "db_path": "session/embeddings.db",
                "user_request": "dame informacion sobre el credito LOC-0004",
                "entity_id": "LOC-0004",
                "chosen_table": "chunks_meta",
                "where_clauses": [],
                "execution": {"ok": True, "row_count": 50, "rows": [[1, "a"]]},
            },
        }
    ]

    text = build_agnostic_user_answer("dame informacion sobre el credito LOC-0004", runs)

    assert "No pude filtrar por LOC-0004" in text
    assert "contabilidad.db" in text


def test_build_agnostic_user_answer_prefers_successful_sql_execution_over_missing_where_clauses():
    runs = [
        {
            "name": "nl2sql",
            "args": {
                "user_request": "Dame la información del crédito LOC-0004 en contabilidad.db",
                "db_path": "contabilidad.db",
                "execute": True,
                "entity_id": "LOC-0004",
            },
            "output": {
                "ok": True,
                "db_path": "session/contabilidad.db",
                "user_request": "Dame la información del crédito LOC-0004 en contabilidad.db",
                "entity_id": "LOC-0004",
                "chosen_table": "estados_cuenta",
                "generated_sql": "SELECT credito_id, cliente_id, estatus, saldo_total FROM estados_cuenta WHERE credito_id = 'LOC-0004' LIMIT 50;",
                "where_clauses": [],
                "execution": {
                    "ok": True,
                    "row_count": 1,
                    "rows": [["LOC-0004", "CLI-0004", "Desembolsado", 29440.64]],
                },
            },
        }
    ]

    text = build_agnostic_user_answer(
        "Dame la información del crédito LOC-0004 en contabilidad.db",
        runs,
    )

    assert "No pude filtrar por LOC-0004" not in text
    assert "Encontré 1 registro" in text
    assert "LOC-0004" in text
