from __future__ import annotations

from typing import Any, Callable, Dict, List

from agnostic_agent.tools.pipeline.catcher_tool import execute_catcher_tool


def execute_catcher_node(
    state: Dict[str, Any],
    *,
    extract_tool_calls: Callable[[Any], List[Dict[str, Any]]],
    decode_tool_content: Callable[[Any], Any],
    to_jsonable: Callable[[Any], Any],
) -> Dict[str, Any]:
    # Structural wrapper: execution logic lives in pipeline tools.
    return execute_catcher_tool(
        state,
        extract_tool_calls=extract_tool_calls,
        decode_tool_content=decode_tool_content,
        to_jsonable=to_jsonable,
    )
