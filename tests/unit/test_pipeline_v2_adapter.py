from agnostic_agent.core.pipeline_v2.adapter import build_pipeline_output_v2, render_deep_text


def test_render_deep_text_uses_summary_v2_only():
    output = build_pipeline_output_v2(
        prompt_text="conciliar",
        out_state={
            "analyzer": {"subqueries": ["q1"], "propositional_logic": "q1"},
            "planner_trajs": [{"subquery": "q1", "description": "step 1: tool=x"}],
            "executor_steps": [{"tool_name": "x", "args": {}, "tool_call_id": "1"}],
            "validator": {"all_covered": True, "reasoning": "ok"},
            "_active_skills_internal": ["s1"],
        },
        summary_obj=None,
        tool_runs=[],
        fallback_final_user="ok",
    )

    text = render_deep_text(output.deep_out)
    assert text.startswith("## Deep Summary")
    assert "### Analyzer" in text
    assert "### Metrics" in text
