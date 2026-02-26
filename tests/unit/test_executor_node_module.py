import json

from langchain_core.messages import AIMessage, ToolMessage

from agnostic_agent.graph.executor_node import execute_executor_node


class _StubTool:
    def __init__(self, name: str, args: dict, output):
        self.name = name
        self.args = args
        self._output = output
        self.invocations = []

    def invoke(self, args):
        self.invocations.append(args)
        return self._output


def _json_default(value):
    return repr(value)


def test_execute_executor_node_runs_and_emits_tool_messages():
    plan = AIMessage(content="", tool_calls=[{"id": "call_1", "name": "reconcile_credit_accounting", "args": {"credito_id": "LOC-1"}}])
    state = {"messages": [plan]}
    tool = _StubTool("reconcile_credit_accounting", {"credito_id": {}}, {"ok": True})

    out = execute_executor_node(
        state,
        tools=[tool],
        ai_message_type=AIMessage,
        tool_message_type=ToolMessage,
        extract_tool_calls=lambda _m: [],
        canonical_tool_name=lambda name: str(name),
        to_jsonable=lambda value: value,
        json_default=_json_default,
    )

    assert len(out["messages"]) == 1
    assert len(out["executor_steps"]) == 1
    assert tool.invocations == [{"credito_id": "LOC-1"}]
    payload = json.loads(out["messages"][0].content)
    assert payload["value"]["ok"] is True


def test_execute_executor_node_supports_dependency_resolution():
    plan = AIMessage(
        content="",
        tool_calls=[
            {"id": "step_1", "name": "tool_a", "args": {"value": 10}},
            {"id": "step_2", "name": "tool_b", "args": {"copied": "$step_1.output"}},
        ],
    )
    state = {"messages": [plan]}
    tool_a = _StubTool("tool_a", {"value": {}}, {"result": "from_a"})
    tool_b = _StubTool("tool_b", {"copied": {}}, {"ok": True})

    out = execute_executor_node(
        state,
        tools=[tool_a, tool_b],
        ai_message_type=AIMessage,
        tool_message_type=ToolMessage,
        extract_tool_calls=lambda _m: [],
        canonical_tool_name=lambda name: str(name),
        to_jsonable=lambda value: value,
        json_default=_json_default,
    )

    assert len(out["executor_steps"]) == 2
    assert tool_b.invocations == [{"copied": {"result": "from_a"}}]


def test_execute_executor_node_repairs_single_arg_name():
    plan = AIMessage(content="", tool_calls=[{"id": "call_1", "name": "tool_one", "args": {"arg_name": "X"}}])
    state = {"messages": [plan]}
    tool = _StubTool("tool_one", {"query": {}}, {"ok": True})

    execute_executor_node(
        state,
        tools=[tool],
        ai_message_type=AIMessage,
        tool_message_type=ToolMessage,
        extract_tool_calls=lambda _m: [],
        canonical_tool_name=lambda name: str(name),
        to_jsonable=lambda value: value,
        json_default=_json_default,
    )

    assert tool.invocations == [{"query": "X"}]
