from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

from agnostic_agent.core.models.io_models import AgentSummary, ToolRun
from agnostic_agent.core.pipeline_v2.contracts import (
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
    deep_vm = DeepViewModelV2(
        timeline=timeline,
        artifacts={
            "summary": summary_obj.model_dump() if summary_obj else {},
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
    lines: List[str] = ["## Deep Pipeline"]
    lines.append("### Timeline")
    for ev in vm.timeline:
        lines.append(f"- {ev.node}: {ev.status} ({ev.duration_ms}ms)")
    lines.append("")
    lines.append("### Artifacts")
    lines.append(_sanitize_text(vm.artifacts))
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
