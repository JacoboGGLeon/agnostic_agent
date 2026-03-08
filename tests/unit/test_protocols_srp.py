from agnostic_agent.protocols.srp import SkillRuntimeRequest, SkillRuntimeResponse


def test_srp_request_defaults():
    req = SkillRuntimeRequest(
        run_id="run_1",
        skill={"name": "semantic_researcher", "version": "0.1.0"},
        goal="answer",
    )
    assert req.protocol == "skill-runtime/v1"
    assert req.inputs == {}
    assert req.context == {}
    assert req.constraints == {}


def test_srp_response_defaults():
    rsp = SkillRuntimeResponse()
    assert rsp.protocol == "skill-runtime/v1"
    assert rsp.status == "success"
    assert rsp.outputs == {}
