from __future__ import annotations

from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage, ToolMessage


def execute_catcher_tool(
    state: Dict[str, Any],
    *,
    extract_tool_calls: Callable[[Any], List[Dict[str, Any]]],
    decode_tool_content: Callable[[Any], Any],
    to_jsonable: Callable[[Any], Any],
) -> Dict[str, Any]:
    messages = state["messages"]

    ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
    ai_plan = next((m for m in reversed(ai_msgs) if extract_tool_calls(m)), None)
    tool_calls = extract_tool_calls(ai_plan) if ai_plan else []

    tmsgs: List[ToolMessage] = [m for m in messages if isinstance(m, ToolMessage)]

    runs: List[Dict[str, Any]] = []
    for tc in tool_calls:
        tm = next((t for t in tmsgs if t.tool_call_id == tc["id"]), None)
        if tm is None:
            continue
        raw = tm.content
        output = decode_tool_content(raw)
        runs.append(
            {
                "id": tc["id"],
                "name": tc["name"],
                "args": tc.get("args", {}) or {},
                "output": to_jsonable(output),
            }
        )

    return {"tool_runs": runs}

