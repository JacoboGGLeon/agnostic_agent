from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Sequence, Set


def compute_invariant_violations(
    *,
    subqueries: Sequence[str],
    logic_form: str,
    planner_trajs: Sequence[Any],
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


def has_coverage_partial(invariant_violations: Sequence[str]) -> bool:
    return any("CoverageInvariant:" in v for v in invariant_violations)


def build_coverage_warning(*, input_object_count: int, runs_count: int) -> str:
    return (
        "Advertencia: cobertura parcial detectada; la respuesta puede estar incompleta "
        f"(objetos_detectados={input_object_count}, tools_ejecutadas={runs_count})."
    )
