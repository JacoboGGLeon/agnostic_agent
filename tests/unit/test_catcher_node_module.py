from langchain_core.messages import AIMessage, ToolMessage

from agnostic_agent.graph.catcher_node import execute_catcher_node


def test_execute_catcher_node_maps_tool_messages_to_runs():
    plan_msg = AIMessage(content="", additional_kwargs={})
    tool_msg = ToolMessage(content='{"ok": true}', tool_call_id="call_1")
    state = {"messages": [plan_msg, tool_msg]}

    def _extract_tool_calls(msg):
        if msg is plan_msg:
            return [{"id": "call_1", "name": "reconcile_credit_accounting", "args": {"credito_id": "LOC-1"}}]
        return []

    out = execute_catcher_node(
        state,
        extract_tool_calls=_extract_tool_calls,
        decode_tool_content=lambda raw: {"raw": raw},
        to_jsonable=lambda value: value,
    )

    assert len(out["tool_runs"]) == 1
    run = out["tool_runs"][0]
    assert run["id"] == "call_1"
    assert run["name"] == "reconcile_credit_accounting"
    assert run["args"]["credito_id"] == "LOC-1"
    assert run["output"]["raw"] == '{"ok": true}'


def test_execute_catcher_node_ignores_missing_tool_message():
    plan_msg = AIMessage(content="", additional_kwargs={})
    state = {"messages": [plan_msg]}

    def _extract_tool_calls(msg):
        if msg is plan_msg:
            return [{"id": "call_missing", "name": "tool_x", "args": {}}]
        return []

    out = execute_catcher_node(
        state,
        extract_tool_calls=_extract_tool_calls,
        decode_tool_content=lambda raw: raw,
        to_jsonable=lambda value: value,
    )

    assert out["tool_runs"] == []
