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


def test_build_agnostic_user_answer_renders_top_n_list_without_collapsing_to_example():
    runs = [
        {
            "name": "nl2sql",
            "args": {
                "user_request": "dame el top 5 creditos con saldo mas alto",
                "db_path": "contabilidad.db",
                "execute": True,
            },
            "output": {
                "ok": True,
                "db_path": "session/contabilidad.db",
                "user_request": "dame el top 5 creditos con saldo mas alto",
                "chosen_table": "estados_cuenta",
                "generated_sql": "SELECT credito_id, saldo_total FROM estados_cuenta ORDER BY saldo_total DESC LIMIT 5",
                "execution": {
                    "ok": True,
                    "row_count": 5,
                    "columns": ["credito_id", "saldo_total"],
                    "rows": [
                        ["LOC-0010", 39336.16],
                        ["LOC-0046", 32967.03],
                        ["LOC-0036", 30462.36],
                        ["LOC-0004", 29440.64],
                        ["LOC-0015", 27648.02],
                    ],
                },
            },
        }
    ]

    text = build_agnostic_user_answer("dame el top 5 creditos con saldo mas alto", runs)

    assert "Top 5" in text
    assert "Ejemplo:" not in text
    assert "1. credito_id=LOC-0010 | saldo_total=39336.16" in text
    assert "5. credito_id=LOC-0015 | saldo_total=27648.02" in text


def test_build_agnostic_user_answer_renders_grouped_aggregate_rows():
    runs = [
        {
            "name": "nl2sql",
            "args": {
                "user_request": "Promedio de monto por tipo en transacciones.db",
                "db_path": "transacciones.db",
                "execute": True,
            },
            "output": {
                "ok": True,
                "db_path": "session/transacciones.db",
                "user_request": "Promedio de monto por tipo en transacciones.db",
                "chosen_table": "movimientos",
                "generated_sql": "SELECT tipo, AVG(monto) AS promedio_monto FROM movimientos GROUP BY tipo LIMIT 50",
                "execution": {
                    "ok": True,
                    "row_count": 3,
                    "columns": ["tipo", "promedio_monto"],
                    "rows": [
                        ["DESEMBOLSO", 12000.0],
                        ["PAGO", 1500.5],
                        ["DESCUENTO", 80.0],
                    ],
                },
            },
        }
    ]

    text = build_agnostic_user_answer("Promedio de monto por tipo en transacciones.db", runs)

    assert "Resultados agregados:" in text
    assert "1. tipo=DESEMBOLSO | promedio_monto=12000.0" in text
    assert "3. tipo=DESCUENTO | promedio_monto=80.0" in text
