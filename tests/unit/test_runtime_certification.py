from agnostic_agent.runtime.certification import assess_skill_maturity


def test_assess_skill_maturity_l1():
    report = assess_skill_maturity(
        skill_name="s1",
        checks={
            "manifest_valid": True,
            "smoke_ok": True,
            "schema_valid": True,
            "errors_normalized": True,
        },
    )
    assert report.level == "L1 Validated"


def test_assess_skill_maturity_l3():
    report = assess_skill_maturity(
        skill_name="s1",
        checks={
            "manifest_valid": True,
            "smoke_ok": True,
            "schema_valid": True,
            "errors_normalized": True,
            "tool_contracts": True,
            "knowledge_contracts": True,
            "artifacts_emitted": True,
            "observability_complete": True,
            "version_stable": True,
        },
    )
    assert report.level == "L3 Production"
