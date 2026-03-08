from __future__ import annotations

from typing import Any, Callable, Dict, List

from agnostic_agent.tools.pipeline.summarizer_tool import execute_summarizer_tool


def execute_summarizer_node(
    state: Dict[str, Any],
    *,
    skill_registry: Any,
    tools: List[Any],
    cfg: Any,
    planner_llm: Any,
    resolve_effective_skills: Callable[[Dict[str, Any], Any], List[str]],
    json_default: Callable[[Any], Any],
    summarize_tool_runs: Callable[[str, List[Dict[str, Any]]], str],
    summarize_tool_runs_compact: Callable[[List[Dict[str, Any]]], str],
    build_user_answer_from_runs: Callable[[str, List[Dict[str, Any]]], str],
    is_technical_answer: Callable[[str], bool],
    find_last_assistant_real: Callable[[List[Any]], Any],
    extract_tool_calls: Callable[[Any], List[Dict[str, Any]]],
    coerce_content_str: Callable[[Any], str],
    strip_think: Callable[[str], str],
) -> Dict[str, Any]:
    # Structural wrapper: execution logic lives in pipeline tools.
    return execute_summarizer_tool(
        state,
        skill_registry=skill_registry,
        tools=tools,
        cfg=cfg,
        planner_llm=planner_llm,
        resolve_effective_skills=resolve_effective_skills,
        json_default=json_default,
        summarize_tool_runs=summarize_tool_runs,
        summarize_tool_runs_compact=summarize_tool_runs_compact,
        build_user_answer_from_runs=build_user_answer_from_runs,
        is_technical_answer=is_technical_answer,
        find_last_assistant_real=find_last_assistant_real,
        extract_tool_calls=extract_tool_calls,
        coerce_content_str=coerce_content_str,
        strip_think=strip_think,
    )
