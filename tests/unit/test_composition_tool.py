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


def test_execute_composition_conditional_then_branch():
    plan = {
        "op": "conditional",
        "condition": {"value": True},
        "then_step": {"skill": "then_skill", "inputs": {"x": 1}},
        "else_step": {"skill": "else_skill", "inputs": {"x": 2}},
    }
    out = execute_composition(plan=plan, invoke_skill=_invoke_skill, run_id="run_cond")
    assert out["status"] == "success"
    assert out["children"][0]["skill"] == "then_skill"


def test_execute_composition_map():
    plan = {
        "op": "map",
        "map_items": [1, 2, 3],
        "map_step": {"skill": "map_skill", "inputs": {"x": 10}},
    }
    out = execute_composition(plan=plan, invoke_skill=_invoke_skill, run_id="run_map")
    assert out["status"] == "success"
    assert len(out["children"]) == 3
    assert len(out["outputs"]) == 3


def test_execute_composition_tree():
    plan = {
        "op": "tree",
        "root": {"skill": "root_skill", "inputs": {"x": 5}},
        "children": [
            {"skill": "child_1", "inputs": {"x": 1}, "inputs_from": "root.outputs"},
            {"skill": "child_2", "inputs": {"x": 2}, "inputs_from": "root.outputs"},
        ],
    }
    out = execute_composition(plan=plan, invoke_skill=_invoke_skill, run_id="run_tree")
    assert out["status"] == "success"
    assert len(out["children"]) == 3
    assert out["children"][0]["skill"] == "root_skill"
