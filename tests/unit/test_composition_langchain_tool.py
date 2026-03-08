from agnostic_agent.tools import get_default_tools


def test_compose_skills_tool_is_registered_and_invocable():
    tools = get_default_tools()
    compose = next((t for t in tools if t.name == "compose_skills"), None)
    assert compose is not None
    out = compose.invoke(
        {
            "plan": {
                "op": "sequential",
                "steps": [{"skill": "demo_skill", "inputs": {"x": 1}}],
            }
        }
    )
    assert out["status"] == "success"
    assert out["op"] == "sequential"
