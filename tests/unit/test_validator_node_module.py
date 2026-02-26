from agnostic_agent.graph.validator_node import execute_validator_node


def test_execute_validator_node_flags_partial_coverage():
    state = {
        "user_prompt": '{"a":1}, {"b":2}',
        "pipeline_summary": {"final_answer": "ok", "summarizer": "algo"},
        "summary": {"final_answer": "ok", "summarizer": "algo"},
        "tool_runs": [{"name": "t1", "args": {}, "output": {"ok": True}}],
        "analyzer": {"subqueries": ["q1", "q2"], "propositional_logic": "q1 AND q2"},
        "planner_trajs": [{"subquery": "q1"}],
        "planner_calls_by_subquery": [
            {"subquery_idx": 1, "subquery": "q1", "planned_calls": 1, "skipped_reason": ""},
            {"subquery_idx": 2, "subquery": "q2", "planned_calls": 0, "skipped_reason": "no_tool_calls_generated"},
        ],
        "executor_steps": [{"tool_name": "t1"}],
        "_planner_scope_internal": {"allowed_tools": ["t1"], "skill_mode": False},
        "messages": [],
    }

    out = execute_validator_node(
        state,
        skill_registry=None,
        resolve_effective_skills=lambda _state, _registry: [],
        is_placeholder_subquery=lambda _s: False,
        env_flag=lambda _name, default=False: default,
        extract_top_level_json_objects=lambda _text: ["{}", "{}"],
        find_last_assistant_real=lambda _msgs: None,
        coerce_content_str=lambda x: str(x) if x is not None else "",
        strip_think=lambda s: s,
        build_user_answer_from_runs=lambda _prompt, _runs: "respuesta reparada",
        is_technical_answer=lambda _text: False,
    )

    assert out["validator"]["all_covered"] is False
    assert "CoverageInvariant:" in out["validator"]["reasoning"]
    assert isinstance(out["coverage_report"], list)
    assert out["coverage_report"][1]["status"] in {"missing", "skipped"}
    assert out["messages"][0].additional_kwargs.get("node") == "validator"


def test_execute_validator_node_flags_semantic_mismatch():
    state = {
        "user_prompt": 'Realiza la conciliación del crédito {"credito_id":"LOC-0010","saldo_total":39336.16}',
        "pipeline_summary": {"final_answer": "ok", "summarizer": "algo"},
        "summary": {"final_answer": "ok", "summarizer": "algo"},
        "tool_runs": [
            {
                "id": "call_s1_abc123",
                "name": "reconcile_credit_accounting",
                "args": {"credito_id": "LOC-0005", "balance": "39336.16"},
                "output": {"ok": True, "credito_id": "LOC-0005"},
            }
        ],
        "analyzer": {
            "subqueries": [
                'Realiza la conciliación del crédito {"credito_id":"LOC-0010","saldo_total":39336.16}'
            ],
            "propositional_logic": "q1",
        },
        "planner_trajs": [{"subquery": "q1"}],
        "planner_calls_by_subquery": [
            {
                "subquery_idx": 1,
                "subquery": "q1",
                "planned_calls": 1,
                "skipped_reason": "",
            }
        ],
        "executor_steps": [
            {
                "tool_name": "reconcile_credit_accounting",
                "args": {"credito_id": "LOC-0005", "balance": "39336.16"},
                "tool_call_id": "call_s1_abc123",
            }
        ],
        "_planner_scope_internal": {"allowed_tools": ["reconcile_credit_accounting"], "skill_mode": False},
        "messages": [],
    }

    out = execute_validator_node(
        state,
        skill_registry=None,
        resolve_effective_skills=lambda _state, _registry: [],
        is_placeholder_subquery=lambda _s: False,
        env_flag=lambda _name, default=False: default,
        extract_top_level_json_objects=lambda _text: ["{}"],
        find_last_assistant_real=lambda _msgs: None,
        coerce_content_str=lambda x: str(x) if x is not None else "",
        strip_think=lambda s: s,
        build_user_answer_from_runs=lambda _prompt, _runs: "respuesta reparada",
        is_technical_answer=lambda _text: False,
    )

    assert out["validator"]["all_covered"] is False
    assert "mismatch semantico" in out["validator"]["reasoning"].lower()
    assert out["coverage_report"][0]["status"] == "mismatch"
