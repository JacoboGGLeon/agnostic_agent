from __future__ import annotations

from typing import Any, Dict, List

from agnostic_agent.tools.pipeline.planner_tool import execute_planner_tool


def execute_planner_node(
    state: Dict[str, Any],
    *,
    tools: List[Any],
    cfg: Any,
    planner_llm: Any,
    skill_registry: Any,
    ai_message_type: Any,
    human_message_type: Any,
    system_message_type: Any,
    planner_trajectory_type: Any,
    resolve_effective_skills: Any,
    is_pipeline_internal_ai: Any,
    is_ai_with_tool_calls: Any,
    strip_think: Any,
    normalize_toolcalls_list: Any,
    extract_tool_calls_from_jsonish_text: Any,
    coerce_content_str: Any,
    canonical_tool_name: Any,
) -> Dict[str, Any]:
    # Structural wrapper: execution logic lives in pipeline tools.
    return execute_planner_tool(
        state,
        tools=tools,
        cfg=cfg,
        planner_llm=planner_llm,
        skill_registry=skill_registry,
        ai_message_type=ai_message_type,
        human_message_type=human_message_type,
        system_message_type=system_message_type,
        planner_trajectory_type=planner_trajectory_type,
        resolve_effective_skills=resolve_effective_skills,
        is_pipeline_internal_ai=is_pipeline_internal_ai,
        is_ai_with_tool_calls=is_ai_with_tool_calls,
        strip_think=strip_think,
        normalize_toolcalls_list=normalize_toolcalls_list,
        extract_tool_calls_from_jsonish_text=extract_tool_calls_from_jsonish_text,
        coerce_content_str=coerce_content_str,
        canonical_tool_name=canonical_tool_name,
    )
