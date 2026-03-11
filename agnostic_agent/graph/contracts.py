from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class AnalyzerResult(TypedDict, total=False):
    input_payload: Dict[str, Any]
    selected_skill_world: str
    selection_mode: str
    propositional_logic: str
    subqueries: List[str]
    subqueries_logic: List[str]
    subquery_intents: List[List[str]]
    entities_by_subquery: List[Dict[str, Any]]
    constraints_by_subquery: List[Dict[str, Any]]
    response_mode: str


class PlannerTrajectory(TypedDict, total=False):
    subquery: str
    description: str
    subquery_id: str
    intent: str
    dag: List[Dict[str, Any]]


class ExecutorStep(TypedDict, total=False):
    node_id: str
    subquery_id: str
    kind: str
    tool_call_id: str
    tool_name: str
    args: Dict[str, Any]
    expected_artifact: str


class SummaryDict(TypedDict, total=False):
    analyzer: str
    planner: str
    executor: str
    catcher: str
    summarizer: str
    final_answer: str


class ValidatorResult(TypedDict, total=False):
    all_covered: bool
    reasoning: str


class State(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    analyzer: Optional[AnalyzerResult]
    planner_trajs: List[PlannerTrajectory]
    planner_calls_by_subquery: List[Dict[str, Any]]
    executor_steps: List[ExecutorStep]
    tool_runs: List[Dict[str, Any]]
    summary: Optional[SummaryDict]
    pipeline_summary: Optional[SummaryDict]
    validator: Optional[ValidatorResult]
    coverage_report: List[Dict[str, Any]]
    artifacts: List[Dict[str, Any]]
    dags_by_subquery: List[Dict[str, Any]]
    selected_skill_world: Optional[str]
    world_contract: Optional[Dict[str, Any]]
    subquery_intents: List[List[str]]
    entities_by_subquery: List[Dict[str, Any]]
    constraints_by_subquery: List[Dict[str, Any]]
    forced_skill: Optional[str]
    skills_allowlist: Optional[List[str]]
    user_prompt: Optional[str]
    session_id: Optional[str]
    knowledge_names: List[str]
    memory_context: Optional[Dict[str, Any]]
    dev_out: Optional[str]
    deep_out: Optional[str]
    user_out: Optional[str]
    llm_raw_out: Optional[str]
    llm_clean_out: Optional[str]
    _active_skills_internal: Optional[List[str]]
    _planner_scope_internal: Optional[Dict[str, Any]]
    _analyzer_skill_selection: Optional[Dict[str, Any]]
