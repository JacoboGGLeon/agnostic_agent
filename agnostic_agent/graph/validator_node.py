from __future__ import annotations

from typing import Any, Callable, Dict, List

from agnostic_agent.tools.pipeline.validator_tool import execute_validator_tool


def execute_validator_node(
    state: Dict[str, Any],
    *,
    skill_registry: Any,
    resolve_effective_skills: Callable[[Dict[str, Any], Any], List[str]],
    is_placeholder_subquery: Callable[[Any], bool],
    env_flag: Callable[[str, bool], bool],
    extract_top_level_json_objects: Callable[[Any], List[str]],
    find_last_assistant_real: Callable[[List[Any]], Any],
    coerce_content_str: Callable[[Any], str],
    strip_think: Callable[[str], str],
    build_user_answer_from_runs: Callable[[str, List[Dict[str, Any]]], str],
    is_technical_answer: Callable[[str], bool],
) -> Dict[str, Any]:
    # Structural wrapper: execution logic lives in pipeline tools.
    return execute_validator_tool(
        state,
        skill_registry=skill_registry,
        resolve_effective_skills=resolve_effective_skills,
        is_placeholder_subquery=is_placeholder_subquery,
        env_flag=env_flag,
        extract_top_level_json_objects=extract_top_level_json_objects,
        find_last_assistant_real=find_last_assistant_real,
        coerce_content_str=coerce_content_str,
        strip_think=strip_think,
        build_user_answer_from_runs=build_user_answer_from_runs,
        is_technical_answer=is_technical_answer,
    )
