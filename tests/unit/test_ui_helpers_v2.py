from agnostic_agent.ui.panels.helpers import extract_summary_deep


def test_extract_summary_deep_prefers_pipeline_v2_bundle():
    raw_state = {
        "pipeline_v2": {
            "deep_out": {
                "summary": {
                    "analyzer": {"subqueries": 2},
                    "planner": {"planned_calls": 3},
                    "metrics": {"coverage_ratio": 1.0},
                }
            }
        }
    }
    text = extract_summary_deep(raw_state, "")
    assert "RESUMEN DEEP DEL PIPELINE (v2)" in text
    assert "### Analyzer" in text
    assert "- subqueries: 2" in text
    assert "### Metrics" in text
