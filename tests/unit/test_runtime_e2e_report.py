from agnostic_agent.runtime.e2e_report import build_end_to_end_report


def test_build_end_to_end_report_contains_protocol_checks():
    report = build_end_to_end_report(
        run_id="run_1",
        prompt_text="hola mundo",
        tool_runs=[],
        protocol_checks={"pipeline_v2_enabled": {"ok": True, "errors": []}},
        user_answer="ok",
    )
    assert report["run_id"] == "run_1"
    assert report["final_answer_non_empty"] is True
    assert "srp_output_shape" in report["protocol_checks"]
