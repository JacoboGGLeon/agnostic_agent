from __future__ import annotations

from typing import Any, Dict, List

from agnostic_agent.tools.pipeline.analyzer_tool import execute_analyzer_tool


def execute_analyzer_node(
    state: Dict[str, Any],
    *,
    tools: List[Any],
    cfg: Any,
    planner_llm: Any,
    skill_registry: Any,
    ai_message_type: Any,
    human_message_type: Any,
    system_message_type: Any,
    coerce_content_str: Any,
    sanitize_subquery_text: Any,
    extract_top_level_json_objects: Any,
    is_placeholder_subquery: Any,
) -> Dict[str, Any]:
    # Structural wrapper: execution logic lives in pipeline tools.
    return execute_analyzer_tool(
        state,
        tools=tools,
        cfg=cfg,
        planner_llm=planner_llm,
        skill_registry=skill_registry,
        ai_message_type=ai_message_type,
        human_message_type=human_message_type,
        system_message_type=system_message_type,
        coerce_content_str=coerce_content_str,
        sanitize_subquery_text=sanitize_subquery_text,
        extract_top_level_json_objects=extract_top_level_json_objects,
        is_placeholder_subquery=is_placeholder_subquery,
    )
