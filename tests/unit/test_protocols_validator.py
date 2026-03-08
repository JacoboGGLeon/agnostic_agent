from agnostic_agent.protocols.validator import (
    validate_kap_contract,
    validate_scp_plan,
    validate_srp_response,
    validate_tcp_contract,
    validate_tep_bundle,
)


def test_validate_scp_plan_ok():
    ok, errors = validate_scp_plan(
        {
            "op": "sequential",
            "steps": [{"skill": "s1", "inputs": {"x": 1}}],
        }
    )
    assert ok is True
    assert errors == []


def test_validate_srp_response_ok():
    ok, errors = validate_srp_response(
        {
            "status": "success",
            "outputs": {},
            "artifacts": [],
            "errors": [],
            "metrics": {},
            "children": [],
        }
    )
    assert ok is True
    assert errors == []


def test_validate_tcp_contract_ok():
    ok, errors = validate_tcp_contract(
        {
            "name": "query_transactions_db",
            "description": "Query tx db",
            "input_schema": "schemas/query.input.json",
            "output_schema": "schemas/query.output.json",
            "side_effects": "read_only",
            "timeout_s": 10,
            "testing": {"mode": "explicit_or_auto"},
        }
    )
    assert ok is True
    assert errors == []


def test_validate_kap_contract_ok():
    ok, errors = validate_kap_contract(
        {
            "name": "finance_kb",
            "description": "kb adapter",
            "entrypoint": "adapters.finance:build",
            "testing": {"mode": "auto"},
        }
    )
    assert ok is True
    assert errors == []


def test_validate_tep_bundle_ok():
    ok, errors = validate_tep_bundle(
        {
            "mode": "auto",
            "records": [
                {
                    "component_type": "skill",
                    "component_name": "contabilidad_instantanea",
                    "check_id": "smoke_execution",
                    "passed": True,
                }
            ],
        }
    )
    assert ok is True
    assert errors == []
