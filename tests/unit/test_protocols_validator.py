from agnostic_agent.protocols.validator import validate_scp_plan, validate_srp_response


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
