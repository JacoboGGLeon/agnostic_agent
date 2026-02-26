from __future__ import annotations

import re
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

from agnostic_agent.core.models.io_models import AgentSummary, ToolRun
from agnostic_agent.core.pipeline_v2.contracts import (
    DeepSummaryV2,
    DeepViewModelV2,
    DevViewModelV2,
    PipelineEvent,
    PipelineOutputV2,
    UserSection,
    UserViewModelV2,
)


def _sanitize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"(?i),?\s*['\"]?\[object\s*object\]['\"]?\s*,?", "", text)
    text = text.replace("[object Object]", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_timeline(
    out_state: Dict[str, Any],
    summary_obj: Optional[AgentSummary],
) -> List[PipelineEvent]:
    analyzer = out_state.get("analyzer") or {}
    planner_trajs = out_state.get("planner_trajs") or []
    executor_steps = out_state.get("executor_steps") or []
    tool_runs = out_state.get("tool_runs") or []
    validator = out_state.get("validator") or {}
    all_covered = bool(validator.get("all_covered", True))

    events: List[PipelineEvent] = [
        PipelineEvent(
            node="analyzer",
            status="ok" if analyzer else "warn",
            payload={"subqueries": len(analyzer.get("subqueries") or [])},
        ),
        PipelineEvent(
            node="planner",
            status="ok" if planner_trajs else "warn",
            payload={"plans": len(planner_trajs)},
        ),
        PipelineEvent(
            node="executor",
            status="ok" if executor_steps else "warn",
            payload={"calls": len(executor_steps)},
        ),
        PipelineEvent(
            node="catcher",
            status="ok" if tool_runs else "warn",
            payload={"tool_runs": len(tool_runs)},
        ),
        PipelineEvent(
            node="summarizer",
            status="ok" if summary_obj and summary_obj.final_answer else "warn",
            payload={"has_final_answer": bool(summary_obj and summary_obj.final_answer)},
        ),
        PipelineEvent(
            node="validator",
            status="ok" if all_covered else "warn",
            payload={"all_covered": all_covered},
        ),
    ]
    return events


def _count_by(values: List[str]) -> Dict[str, int]:
    if not values:
        return {}
    return dict(Counter(v for v in values if v))


def _extract_planned_tools(planner_trajs: List[Any]) -> List[str]:
    planned: List[str] = []
    for traj in planner_trajs:
        description = ""
        if isinstance(traj, dict):
            description = str(traj.get("description", "") or "")
        else:
            description = str(getattr(traj, "description", "") or "")
        planned.extend(re.findall(r"tool=([a-zA-Z0-9_]+)", description))
    return planned


def _build_deep_summary_v2(
    *,
    out_state: Dict[str, Any],
    tool_runs: List[ToolRun],
    findings: List[str],
    final_answer: str,
) -> DeepSummaryV2:
    analyzer = out_state.get("analyzer") or {}
    planner_trajs = out_state.get("planner_trajs") or []
    executor_steps = out_state.get("executor_steps") or []
    validator = out_state.get("validator") or {}
    analyzer_subqueries = analyzer.get("subqueries") or []

    planned_tools = _extract_planned_tools(planner_trajs)
    executed_tools = [
        str(step.get("tool_name", "") or "")
        for step in executor_steps
        if isinstance(step, dict)
    ]
    run_tools = [str(run.name or "") for run in tool_runs]
    output_types = _count_by([type(run.output).__name__ for run in tool_runs])
    final_output: Dict[str, Any] = {}
    if tool_runs:
        last_run = tool_runs[-1]
        final_output = {
            "id": _sanitize_text(last_run.id),
            "tool": _sanitize_text(last_run.name),
            "input": last_run.args if isinstance(last_run.args, dict) else {},
            "output": last_run.output,
        }
    all_tool_outputs: List[Dict[str, Any]] = []
    for run in tool_runs:
        all_tool_outputs.append(
            {
                "id": _sanitize_text(run.id),
                "tool": _sanitize_text(run.name),
                "input": run.args if isinstance(run.args, dict) else {},
                "output": run.output,
            }
        )

    subqueries = len(analyzer.get("subqueries") or [])
    planned_calls = len(planned_tools)
    executed_calls = len(executed_tools)
    run_count = len(tool_runs)
    coverage_ratio = 0.0
    coverage_report = out_state.get("coverage_report") or []
    if isinstance(coverage_report, list) and coverage_report:
        covered = 0
        total = 0
        for row in coverage_report:
            if not isinstance(row, dict):
                continue
            total += 1
            if str(row.get("status", "")) == "executed":
                covered += 1
        if total > 0:
            coverage_ratio = round(covered / total, 3)
    elif subqueries > 0:
        coverage_ratio = round(min(1.0, run_count / subqueries), 3)

    return DeepSummaryV2(
        analyzer={
            "subqueries": subqueries,
            "logic": _sanitize_text(analyzer.get("propositional_logic", "")),
            "active_skills": out_state.get("_active_skills_internal") or [],
            "subquery_rows": [
                {"idx": i, "subquery": _sanitize_text(sq)}
                for i, sq in enumerate(analyzer_subqueries, start=1)
            ],
        },
        planner={
            "subqueries_planned": len(planner_trajs),
            "planned_calls": planned_calls,
            "calls_by_tool": _count_by(planned_tools),
        },
        executor={
            "executed_calls": executed_calls,
            "calls_by_tool": _count_by(executed_tools),
        },
        catcher={
            "tool_runs": run_count,
            "runs_by_tool": _count_by(run_tools),
            "output_types": output_types,
        },
        summarizer={
            "findings": len(findings),
            "final_answer_chars": len(final_answer),
        },
        validator={
            "all_covered": bool(validator.get("all_covered", True)),
            "reasoning": _sanitize_text(validator.get("reasoning", "")),
            "coverage_report": coverage_report if isinstance(coverage_report, list) else [],
        },
        final_output=final_output,
        tool_outputs={"runs": all_tool_outputs},
        metrics={
            "subqueries": subqueries,
            "planned_calls": planned_calls,
            "executed_calls": executed_calls,
            "tool_runs": run_count,
            "coverage_ratio": coverage_ratio,
        },
    )


def build_pipeline_output_v2(
    *,
    prompt_text: str,
    out_state: Dict[str, Any],
    summary_obj: Optional[AgentSummary],
    tool_runs: List[ToolRun],
    fallback_final_user: str,
) -> PipelineOutputV2:
    final_answer = _sanitize_text(
        (summary_obj.final_answer if summary_obj else "") or fallback_final_user
    )

    warnings: List[str] = []
    validator = out_state.get("validator") or {}
    if validator and not bool(validator.get("all_covered", True)):
        warnings.append(_sanitize_text(validator.get("reasoning") or "Cobertura parcial detectada."))
    if not final_answer:
        warnings.append("No se obtuvo respuesta final no vacia.")

    findings: List[str] = []
    for run in tool_runs:
        name = _sanitize_text(run.name)
        args = run.args if isinstance(run.args, dict) else {}
        output = run.output if isinstance(run.output, dict) else {}
        entity = ""
        for key in ("id", "entity_id", "record_id", "item_id", "document_id", "ticket_id", "task_id"):
            if args.get(key) not in (None, ""):
                entity = f"{key}={args.get(key)}"
                break
        if not entity:
            for key, value in args.items():
                if str(key).endswith("_id") and value not in (None, ""):
                    entity = f"{key}={value}"
                    break
        if isinstance(output, dict):
            status = output.get("status") or output.get("result") or ("ok=true" if output.get("ok") else "resultado=estructurado")
            if output.get("error"):
                status = f"error={output.get('error')}"
        else:
            status = f"resultado={type(run.output).__name__}"
        findings.append(f"{name}{f' ({entity})' if entity else ''}: {status}")

    user_vm = UserViewModelV2(
        final_answer=final_answer,
        sections=[
            UserSection(title="Solicitud", items=[_sanitize_text(prompt_text)]),
            UserSection(title="Hallazgos", items=findings),
        ],
        warnings=warnings,
    )

    timeline = _build_timeline(out_state, summary_obj)
    summary_v2 = _build_deep_summary_v2(
        out_state=out_state,
        tool_runs=tool_runs,
        findings=findings,
        final_answer=final_answer,
    )
    deep_vm = DeepViewModelV2(
        timeline=timeline,
        summary=summary_v2,
        artifacts={
            "summary_v2": summary_v2.model_dump(),
            "tool_runs_count": len(tool_runs),
        },
        raw={
            "analyzer": out_state.get("analyzer"),
            "planner_trajs": out_state.get("planner_trajs"),
            "executor_steps": out_state.get("executor_steps"),
            "validator": out_state.get("validator"),
        },
    )

    dev_vm = DevViewModelV2(
        summary="Pipeline v2 output built from typed artifacts.",
        counts={
            "subqueries": len((out_state.get("analyzer") or {}).get("subqueries") or []),
            "tool_calls": len(out_state.get("executor_steps") or []),
            "tool_runs": len(tool_runs),
        },
        issues=warnings,
    )

    return PipelineOutputV2(
        user_out=user_vm,
        deep_out=deep_vm,
        dev_out=dev_vm,
        turn_id=str(uuid.uuid4()),
    )


def render_user_text(vm: UserViewModelV2) -> str:
    lines: List[str] = []
    if vm.final_answer:
        lines.append(vm.final_answer)
    for section in vm.sections:
        if not section.items:
            continue
        lines.append("")
        lines.append(f"### {section.title}")
        lines.extend(f"- {item}" for item in section.items)
    if vm.warnings:
        lines.append("")
        lines.append("### Warnings")
        lines.extend(f"- {w}" for w in vm.warnings)
    return _sanitize_text("\n".join(lines))


def render_deep_text(vm: DeepViewModelV2) -> str:
    lines: List[str] = ["## Deep Summary"]
    summary = vm.summary.model_dump() if vm.summary else {}
    if not isinstance(summary, dict) or not summary:
        summary = vm.artifacts.get("summary_v2", {}) if isinstance(vm.artifacts, dict) else {}

    if not isinstance(summary, dict) or not summary:
        return _sanitize_text("\n".join(lines + ["- (no summary)"]))

    section_order = [
        ("Analyzer", "analyzer"),
        ("Planner", "planner"),
        ("Executor", "executor"),
        ("Catcher", "catcher"),
        ("Summarizer", "summarizer"),
        ("Validator", "validator"),
        ("Final Output", "final_output"),
        ("Tool Outputs", "tool_outputs"),
        ("Metrics", "metrics"),
    ]
    for title, key in section_order:
        section = summary.get(key) or {}
        if not isinstance(section, dict) or not section:
            continue
        lines.append("")
        lines.append(f"### {title}")
        for field, value in section.items():
            lines.append(f"- {field}: {_sanitize_text(value)}")
    return _sanitize_text("\n".join(lines))


def render_dev_text(vm: DevViewModelV2) -> str:
    lines = [
        "## Dev Summary",
        vm.summary,
        f"Counts: {vm.counts}",
    ]
    if vm.issues:
        lines.append("Issues:")
        lines.extend(f"- {i}" for i in vm.issues)
    return _sanitize_text("\n".join(lines))
