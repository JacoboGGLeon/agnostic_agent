from agnostic_agent.protocols.tep import TEPBundle, validate_tep_minimum_checks


def test_tep_bundle_defaults():
    bundle = TEPBundle()
    assert bundle.protocol == "test-evidence/v1"
    assert bundle.mode == "explicit_or_auto"
    assert bundle.records == []


def test_tep_minimum_checks_ok_for_single_skill():
    bundle = TEPBundle(
        mode="auto",
        records=[
            {
                "component_type": "skill",
                "component_name": "contabilidad_instantanea",
                "check_id": "manifest_validation",
                "passed": True,
            },
            {
                "component_type": "skill",
                "component_name": "contabilidad_instantanea",
                "check_id": "smoke_execution",
                "passed": True,
            },
            {
                "component_type": "skill",
                "component_name": "contabilidad_instantanea",
                "check_id": "input_schema_validation",
                "passed": True,
            },
            {
                "component_type": "skill",
                "component_name": "contabilidad_instantanea",
                "check_id": "output_schema_validation",
                "passed": True,
            },
            {
                "component_type": "skill",
                "component_name": "contabilidad_instantanea",
                "check_id": "error_normalization",
                "passed": True,
            },
        ],
    )
    ok, missing = validate_tep_minimum_checks(bundle)
    assert ok is True
    assert missing == []


def test_tep_minimum_checks_detects_missing():
    bundle = TEPBundle(
        mode="auto",
        records=[
            {
                "component_type": "tool",
                "component_name": "query_transactions_db",
                "check_id": "importability",
                "passed": True,
            },
        ],
    )
    ok, missing = validate_tep_minimum_checks(bundle)
    assert ok is False
    assert any("schema_conformance" in item for item in missing)
