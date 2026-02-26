from agnostic_agent.ui.panels.helpers import extract_summary_deep


def test_extract_summary_deep_prefers_pipeline_v2_bundle():
    raw_state = {
        "pipeline_v2": {
            "deep_out": {
                "timeline": [
                    {"node": "analyzer", "status": "ok", "duration_ms": 0},
                    {"node": "planner", "status": "ok", "duration_ms": 0},
                ],
                "artifacts": {"tool_runs_count": 2},
            }
        }
    }
    text = extract_summary_deep(raw_state, "")
    assert "RESUMEN DEEP DEL PIPELINE (v2)" in text
    assert "analyzer: ok" in text
    assert '"tool_runs_count": 2' in text
