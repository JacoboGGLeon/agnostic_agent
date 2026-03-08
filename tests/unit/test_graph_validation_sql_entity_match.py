from agnostic_agent.graph.validation import build_subquery_coverage_report


def test_subquery_coverage_report_matches_entity_from_sql_query_args():
    report = build_subquery_coverage_report(
        subqueries=["Realiza la conciliacion del Credito LOC-0005"],
        planner_calls_by_subquery=[{"subquery_idx": 1, "planned_calls": 2, "skipped_reason": ""}],
        executor_steps=[
            {
                "tool_name": "query_accounting_db",
                "args": {"query": "SELECT saldo_total FROM estados_cuenta WHERE credito_id = 'LOC-0005'"},
                "tool_call_id": "call_s1_abc123",
            }
        ],
        tool_runs=[
            {
                "id": "call_s1_abc123",
                "name": "query_accounting_db",
                "args": {"query": "SELECT saldo_total FROM estados_cuenta WHERE credito_id = 'LOC-0005'"},
                "output": '{"normalized_query":"SELECT saldo_total FROM estados_cuenta WHERE credito_id = \'LOC-0005\'", "rows": [[1]]}',
            }
        ],
    )

    assert report[0]["status"] == "executed"
    assert report[0]["skipped_reason"] == ""
