from agnostic_agent.core.models.runtime_objects import (
    Action,
    Artifact,
    KnowledgeItem,
    ProviderResponse,
    RunContext,
    SkillRequest,
    SkillResult,
    ToolCall,
    ToolResult,
)


def test_runtime_objects_minimal_shapes():
    ctx = RunContext(run_id="run_1")
    act = Action(type="respond", payload={"text": "ok"})
    art = Artifact(artifact_id="a1", run_id="run_1", kind="trace", producer="runtime")
    req = SkillRequest(skill_name="s1", goal="g")
    res = SkillResult(status="success")
    tc = ToolCall(name="t1")
    tr = ToolResult(ok=True, data={"x": 1})
    ki = KnowledgeItem(id="k1", type="chunk", content="c", source="src")
    pr = ProviderResponse(text="ok")

    assert ctx.run_id == "run_1"
    assert act.type == "respond"
    assert art.producer == "runtime"
    assert req.skill_name == "s1"
    assert res.status == "success"
    assert tc.name == "t1"
    assert tr.ok is True
    assert ki.id == "k1"
    assert pr.text == "ok"
