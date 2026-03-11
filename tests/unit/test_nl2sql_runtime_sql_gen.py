from agnostic_agent.tools.nl2sql_runtime.sql_gen import generate_sql


def test_generate_sql_heuristic_uses_relationship_payload_for_join():
    plan = {"nodes": []}
    outputs = {
        "step1": {
            "type": "table_payload",
            "primary": {"table": "estados_cuenta"},
            "candidates": [{"table": "estados_cuenta"}],
        },
        "step2": {
            "type": "columns_payload",
            "candidates": [
                {"table": "estados_cuenta", "column": "credito_id"},
                {"table": "estados_cuenta", "column": "estatus"},
                {"table": "movimientos", "column": "monto"},
            ],
        },
        "step3": {
            "type": "relationship_payload",
            "primary": {
                "rich_context": {
                    "left_table": "estados_cuenta",
                    "left_column": "credito_id",
                    "right_table": "movimientos",
                    "right_column": "credito_id",
                }
            },
            "candidates": [],
        },
    }

    out = generate_sql(
        user_query="cuantos movimientos por estatus relaciona estados_cuenta con movimientos",
        plan=plan,
        outputs=outputs,
        row_limit=25,
        entity_id="LOC-0004",
    )

    sql = str(out.get("sql_proposal", "")).lower()
    assert " join " in sql
    assert "on t1.credito_id = t2.credito_id" in sql
    assert "group by estatus" in sql
    assert "t1.credito_id = 'loc-0004'" in sql
