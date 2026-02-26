from langchain_core.messages import AIMessage

from agnostic_agent.graph.builders import route_from_planner


def test_route_from_planner_uses_native_tool_calls():
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{"id": "call_1", "name": "x", "args": {}}])
        ]
    }
    route = route_from_planner(
        state,
        ai_message_type=AIMessage,
        extract_tool_calls=lambda _m: [],
    )
    assert route == "executor"


def test_route_from_planner_uses_extracted_calls():
    state = {"messages": [AIMessage(content="tool_uses in text")]}
    route = route_from_planner(
        state,
        ai_message_type=AIMessage,
        extract_tool_calls=lambda _m: [{"name": "x", "args": {}}],
    )
    assert route == "executor"


def test_route_from_planner_goes_to_summarizer_without_calls():
    state = {"messages": [AIMessage(content="plain text")]}
    route = route_from_planner(
        state,
        ai_message_type=AIMessage,
        extract_tool_calls=lambda _m: [],
    )
    assert route == "summarizer"
