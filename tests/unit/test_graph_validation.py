from agnostic_agent.graph.validation import (
    build_subquery_coverage_report,
    build_coverage_warning,
    compute_invariant_violations,
    has_coverage_partial,
)


def test_compute_invariant_violations_detects_coverage_gap():
    violations = compute_invariant_violations(
        subqueries=["q1 task", "q2 task"],
        logic_form="q1 AND q2",
        planner_trajs=[{"subquery": "q1 task"}],
        planner_calls_by_subquery=[{"subquery_idx": 1, "planned_calls": 1}, {"subquery_idx": 2, "planned_calls": 0}],
        executor_steps=[{"tool_name": "t1"}],
        runs_count=1,
        input_object_count=2,
        active_skills_eff=[],
        planner_scope={},
        is_placeholder_subquery=lambda _: False,
    )

    assert any("CoverageInvariant:" in v for v in violations)
    assert has_coverage_partial(violations) is True


def test_build_coverage_warning_contains_counts():
    warning = build_coverage_warning(input_object_count=5, runs_count=3)
    assert "objetos_detectados=5" in warning
    assert "tools_ejecutadas=3" in warning


def test_build_subquery_coverage_report_marks_semantic_mismatch():
    report = build_subquery_coverage_report(
        subqueries=[
            'Realiza la conciliación del crédito {"credito_id":"LOC-0010","saldo_total":39336.16}'
        ],
        planner_calls_by_subquery=[
            {"subquery_idx": 1, "subquery": "q1", "planned_calls": 1, "skipped_reason": ""}
        ],
        executor_steps=[
            {
                "tool_name": "reconcile_credit_accounting",
                "args": {"credito_id": "LOC-0005", "balance": "39336.16"},
                "tool_call_id": "call_s1_abc123",
            }
        ],
        tool_runs=[
            {
                "id": "call_s1_abc123",
                "name": "reconcile_credit_accounting",
                "args": {"credito_id": "LOC-0005", "balance": "39336.16"},
                "output": {"credito_id": "LOC-0005", "ok": True},
            }
        ],
    )

    assert report[0]["status"] == "mismatch"
    assert "entity_mismatch" in report[0]["skipped_reason"]
