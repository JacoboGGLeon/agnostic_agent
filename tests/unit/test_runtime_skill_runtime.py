from agnostic_agent.runtime.skill_runtime import invoke_skill_srp


def _invoke_ok(skill_name, inputs):
    return {"status": "success", "outputs": {"skill": skill_name, "echo": inputs}}


def test_invoke_skill_srp_ok():
    out = invoke_skill_srp(
        request_payload={
            "run_id": "run_1",
            "skill": {"name": "demo_skill", "version": "0.1.0"},
            "goal": "test",
            "inputs": {"x": 1},
            "context": {},
            "constraints": {},
        },
        invoke_skill_impl=_invoke_ok,
    )
    assert out["status"] == "success"
    assert out["outputs"]["skill"] == "demo_skill"


def test_invoke_skill_srp_normalizes_non_dict_result():
    out = invoke_skill_srp(
        request_payload={
            "run_id": "run_1",
            "skill": {"name": "demo_skill"},
            "goal": "test",
            "inputs": {},
            "context": {},
            "constraints": {},
        },
        invoke_skill_impl=lambda _n, _i: "bad",
    )
    assert out["status"] == "error"
    assert out["errors"][0]["code"] == "INVALID_SKILL_RESULT"
