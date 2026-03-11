from __future__ import annotations

from typing import Any, Dict, Set


_NODE_INPUT_REQUIRED: Dict[str, Set[str]] = {
    "analyzer": {"messages"},
    "planner": {"messages", "analyzer"},
    "executor": {"messages"},
    "catcher": {"messages"},
    "summarizer": {"messages"},
    "validator": {"messages"},
}

_NODE_OUTPUT_ALLOWED: Dict[str, Set[str]] = {
    "analyzer": {
        "analyzer",
        "_active_skills_internal",
        "_analyzer_skill_selection",
        "messages",
        "selected_skill_world",
        "subquery_intents",
        "entities_by_subquery",
        "required_sources_by_subquery",
        "constraints_by_subquery",
        "world_contract",
    },
    "planner": {
        "messages",
        "planner_trajs",
        "planner_calls_by_subquery",
        "dags_by_subquery",
        "llm_raw_out",
        "llm_clean_out",
        "_planner_scope_internal",
        "selected_skill_world",
        "world_contract",
    },
    "executor": {"messages", "executor_steps", "artifacts"},
    "catcher": {"tool_runs", "artifacts"},
    "summarizer": {"messages", "summary", "pipeline_summary", "dev_out", "deep_out", "user_out"},
    "validator": {
        "validator",
        "coverage_report",
        "messages",
        "pipeline_summary",
        "summary",
        "user_out",
    },
}


def validate_node_input(node_name: str, state: Dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise TypeError(f"{node_name} input must be dict, got {type(state).__name__}")
    required = _NODE_INPUT_REQUIRED.get(node_name, set())
    missing = sorted(k for k in required if k not in state)
    if missing:
        raise ValueError(f"{node_name} input missing required keys: {missing}")


def validate_node_output(node_name: str, delta: Dict[str, Any]) -> None:
    if not isinstance(delta, dict):
        raise TypeError(f"{node_name} output must be dict, got {type(delta).__name__}")
    allowed = _NODE_OUTPUT_ALLOWED.get(node_name, set())
    extra = sorted(k for k in delta.keys() if k not in allowed)
    if extra:
        raise ValueError(f"{node_name} output contains non-contract keys: {extra}")
