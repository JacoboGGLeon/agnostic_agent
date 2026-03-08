from __future__ import annotations

from typing import Any, Callable, Dict, List

from agnostic_agent.tools.pipeline.executor_tool import execute_executor_tool


def execute_executor_node(
    state: Dict[str, Any],
    *,
    tools: List[Any],
    ai_message_type: Any,
    tool_message_type: Any,
    extract_tool_calls: Callable[[Any], List[Dict[str, Any]]],
    canonical_tool_name: Callable[[Any], str],
    to_jsonable: Callable[[Any], Any],
    json_default: Callable[[Any], Any],
) -> Dict[str, Any]:
    # Structural wrapper: execution logic lives in pipeline tools.
    return execute_executor_tool(
        state,
        tools=tools,
        ai_message_type=ai_message_type,
        tool_message_type=tool_message_type,
        extract_tool_calls=extract_tool_calls,
        canonical_tool_name=canonical_tool_name,
        to_jsonable=to_jsonable,
        json_default=json_default,
    )
