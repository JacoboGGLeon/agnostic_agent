from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


def route_from_planner(
    state: dict,
    *,
    ai_message_type: type,
    extract_tool_calls: Callable[[Any], list],
) -> str:
    messages = state.get("messages", []) or []
    ai_msgs = [m for m in messages if isinstance(m, ai_message_type)]
    if not ai_msgs:
        return "summarizer"

    last_ai = ai_msgs[-1]

    tc = getattr(last_ai, "tool_calls", None)
    if tc and isinstance(tc, list) and len(tc) > 0:
        logger.debug("router -> executor (native tool_calls=%s)", len(tc))
        return "executor"

    extracted = extract_tool_calls(last_ai)
    if extracted:
        logger.debug("router -> executor (extracted tool_calls=%s)", len(extracted))
        return "executor"

    preview = str(getattr(last_ai, "content", ""))[:50]
    logger.debug("router -> summarizer (no tool calls; preview=%s)", preview)
    return "summarizer"


def compile_agent_graph(
    state_type: Any,
    *,
    analyzer_node: Callable[[dict], dict],
    planner_node: Callable[[dict], dict],
    executor_node: Callable[[dict], dict],
    catcher_node: Callable[[dict], dict],
    summarizer_node: Callable[[dict], dict],
    validator_node: Callable[[dict], dict],
    route_from_planner_fn: Callable[[dict], str],
) -> Any:
    builder = StateGraph(state_type)

    builder.add_node("analyzer", analyzer_node)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("catcher", catcher_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("validator", validator_node)

    builder.add_edge(START, "analyzer")
    builder.add_edge("analyzer", "planner")
    builder.add_conditional_edges(
        "planner",
        route_from_planner_fn,
        ["executor", "summarizer"],
    )
    builder.add_edge("executor", "catcher")
    builder.add_edge("catcher", "summarizer")
    builder.add_edge("summarizer", "validator")
    builder.add_edge("validator", END)

    return builder.compile()
