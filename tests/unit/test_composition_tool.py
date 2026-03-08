from agnostic_agent.tools.composition import execute_composition


def _invoke_skill(skill_name, inputs):
    return {
        "status": "success",
        "outputs": {"skill": skill_name, "value": inputs.get("x", 0) + 1},
    }


def test_execute_composition_sequential():
    plan = {
        "op": "sequential",
        "steps": [
            {"skill": "s1", "inputs": {"x": 1}},
            {"skill": "s2", "inputs": {"x": 2}},
        ],
    }
    out = execute_composition(plan=plan, invoke_skill=_invoke_skill, run_id="run_seq")
    assert out["status"] == "success"
    assert out["op"] == "sequential"
    assert len(out["children"]) == 2
    assert out["artifacts"][-1]["kind"] == "run.completed"


def test_execute_composition_parallel():
    plan = {
        "op": "parallel",
        "steps": [
            {"skill": "s1", "inputs": {"x": 1}},
            {"skill": "s2", "inputs": {"x": 2}},
        ],
    }
    out = execute_composition(plan=plan, invoke_skill=_invoke_skill, run_id="run_par")
    assert out["status"] == "success"
    assert out["op"] == "parallel"
    assert len(out["children"]) == 2
    assert "s1" in out["outputs"] and "s2" in out["outputs"]
