from pathlib import Path

from agnostic_agent.runtime.certification import assess_skill_maturity
from agnostic_agent.runtime.tep_store import append_tep_report, load_tep_reports


def test_append_and_load_tep_report(tmp_path: Path):
    path = tmp_path / "tep_reports.json"
    report = assess_skill_maturity(
        skill_name="s1",
        checks={
            "manifest_valid": True,
            "smoke_ok": True,
            "schema_valid": True,
            "errors_normalized": True,
        },
    )
    append_tep_report(path, report)
    data = load_tep_reports(path)
    assert len(data) == 1
    assert data[0]["skill_name"] == "s1"
