from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Sequence, Set


def compute_invariant_violations(
    *,
    subqueries: Sequence[str],
    logic_form: str,
    planner_trajs: Sequence[Any],
    planner_calls_by_subquery: Sequence[Dict[str, Any]],
    executor_steps: Sequence[Dict[str, Any]],
    runs_count: int,
    input_object_count: int,
    active_skills_eff: Sequence[str],
    planner_scope: Dict[str, Any],
    is_placeholder_subquery: Callable[[Any], bool],
) -> List[str]:
    invariant_violations: List[str] = []

    scoped_tools: Set[str] = set(planner_scope.get("allowed_tools") or [])

    if not subqueries:
        invariant_violations.append(
            "DecompositionInvariant: Analyzer no produjo subqueries."
        )
    if logic_form and not re.search(r"\bq1\b", logic_form):
        invariant_violations.append(
            "LogicInvariant: lógica proposicional no contiene al menos q1."
        )
    if subqueries and len(planner_trajs) < len(subqueries):
        invariant_violations.append(
            "CoverageInvariant: planner_trajs cubre menos subqueries que Analyzer."
        )
    if subqueries and len(executor_steps) < len(subqueries) and runs_count > 0:
        invariant_violations.append(
            "CoverageInvariant: executor_steps cubre menos subqueries que Analyzer."
        )
    if subqueries and planner_calls_by_subquery:
        missing_plans = [
            row
            for row in planner_calls_by_subquery
            if int(row.get("planned_calls", 0) or 0) == 0
        ]
        if missing_plans and len(subqueries) > 1:
            invariant_violations.append(
                "CoverageInvariant: existen subqueries sin llamadas planificadas."
            )
    if input_object_count > 1 and runs_count < input_object_count:
        invariant_violations.append(
            "CoverageInvariant: el prompt contiene "
            f"{input_object_count} objetos JSON y solo se ejecutaron {runs_count} tools."
        )
    if subqueries and all(is_placeholder_subquery(s) for s in subqueries):
        invariant_violations.append(
            "DecompositionInvariant: subqueries quedaron en placeholders (q1/paso/pregunta)."
        )

    if active_skills_eff and not planner_scope.get("skill_mode"):
        invariant_violations.append(
            "SkillScopeInvariant: hay skills activas efectivas pero Planner no activó skill_mode."
        )
    if scoped_tools:
        for step in executor_steps:
            tname = step.get("tool_name")
            if tname and tname not in scoped_tools:
                invariant_violations.append(
                    f"ToolScopeInvariant: tool ejecutada fuera de scope permitido ({tname})."
                )

    return invariant_violations


def build_subquery_coverage_report(
    *,
    subqueries: Sequence[str],
    planner_calls_by_subquery: Sequence[Dict[str, Any]],
    executor_steps: Sequence[Dict[str, Any]],
    tool_runs: Sequence[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []
    executed_by_idx: Dict[int, int] = {}
    for step in executor_steps:
        tcid = str(step.get("tool_call_id", "") or "")
        m = re.match(r"^call_s(\d+)_", tcid)
        if not m:
            continue
        idx = int(m.group(1))
        executed_by_idx[idx] = executed_by_idx.get(idx, 0) + 1

    planner_map: Dict[int, Dict[str, Any]] = {}
    for row in planner_calls_by_subquery or []:
        idx = int(row.get("subquery_idx", 0) or 0)
        if idx > 0:
            planner_map[idx] = row

    for idx, subq in enumerate(subqueries or [], start=1):
        row = planner_map.get(idx, {})
        planned = int(row.get("planned_calls", 0) or 0)
        executed = int(executed_by_idx.get(idx, 0))
        skipped_reason = str(row.get("skipped_reason", "") or "")
        status = "missing"
        if executed > 0:
            status = "executed"
        elif skipped_reason:
            status = "skipped"
        semantic = _semantic_match_status(
            subquery=str(subq),
            subquery_idx=idx,
            executor_steps=executor_steps,
            tool_runs=tool_runs or [],
        )
        if status == "executed" and semantic == "mismatch":
            status = "mismatch"
            skipped_reason = "entity_mismatch_between_subquery_and_tool_args"
        report.append(
            {
                "subquery_idx": idx,
                "subquery": str(subq),
                "planned_calls": planned,
                "executed_calls": executed,
                "skipped_reason": skipped_reason,
                "status": status,
            }
        )
    return report


def _extract_expected_id_values(text: str) -> Set[str]:
    values: Set[str] = set()
    source = str(text or "")
    for chunk in re.findall(r"\{[^{}]+\}", source):
        try:
            import json

            obj = json.loads(chunk)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for key, value in obj.items():
            if str(key).endswith("_id") and value not in (None, ""):
                values.add(str(value).strip().lower())
    if values:
        return values
    m = re.search(r"\b[A-Za-z]{2,}[A-Za-z0-9_-]*-[A-Za-z0-9_-]+\b", source)
    if m:
        values.add(m.group(0).strip().lower())
    return values


def _extract_run_id_values(step: Dict[str, Any], run: Dict[str, Any]) -> Set[str]:
    values: Set[str] = set()
    args = step.get("args", {}) if isinstance(step, dict) else {}
    if isinstance(args, dict):
        for key, value in args.items():
            if str(key).endswith("_id") and value not in (None, ""):
                values.add(str(value).strip().lower())
    output = run.get("output", {}) if isinstance(run, dict) else {}
    if isinstance(output, dict):
        for key, value in output.items():
            if str(key).endswith("_id") and value not in (None, ""):
                values.add(str(value).strip().lower())
    return values


def _semantic_match_status(
    *,
    subquery: str,
    subquery_idx: int,
    executor_steps: Sequence[Dict[str, Any]],
    tool_runs: Sequence[Dict[str, Any]],
) -> str:
    expected = _extract_expected_id_values(subquery)
    if not expected:
        return "unknown"

    matched = False
    seen = False
    runs_by_call_id: Dict[str, Dict[str, Any]] = {}
    for run in tool_runs or []:
        call_id = str(run.get("id", "") or "")
        if call_id:
            runs_by_call_id[call_id] = run

    for step in executor_steps or []:
        tcid = str(step.get("tool_call_id", "") or "")
        m = re.match(r"^call_s(\d+)_", tcid)
        if not m or int(m.group(1)) != subquery_idx:
            continue
        seen = True
        run = runs_by_call_id.get(tcid, {})
        observed = _extract_run_id_values(step, run)
        if observed and any(val in expected for val in observed):
            matched = True
            break

    if not seen:
        return "unknown"
    return "match" if matched else "mismatch"


def has_coverage_partial(invariant_violations: Sequence[str]) -> bool:
    return any("CoverageInvariant:" in v for v in invariant_violations)


def build_coverage_warning(*, input_object_count: int, runs_count: int) -> str:
    return (
        "Advertencia: cobertura parcial detectada; la respuesta puede estar incompleta "
        f"(objetos_detectados={input_object_count}, tools_ejecutadas={runs_count})."
    )
