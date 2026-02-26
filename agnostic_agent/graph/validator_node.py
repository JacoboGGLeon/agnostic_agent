from __future__ import annotations

from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage

from agnostic_agent.graph.validation import (
    build_coverage_warning,
    compute_invariant_violations,
    has_coverage_partial,
)


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
) -> Dict[str, Any]:
    user_prompt = state.get("user_prompt") or ""
    summary = state.get("pipeline_summary") or state.get("summary") or {}
    final_answer = summary.get("final_answer") or ""
    summarizer_text = summary.get("summarizer") or ""
    runs = state.get("tool_runs", []) or []
    analyzer = state.get("analyzer") or {}
    planner_trajs = state.get("planner_trajs", []) or []
    executor_steps = state.get("executor_steps", []) or []
    planner_scope = state.get("_planner_scope_internal") or {}
    fail_fast = env_flag("AGNOSTIC_FAIL_FAST", False)
    input_object_count = len(extract_top_level_json_objects(user_prompt))

    bad_templates = (
        "no se invoco ninguna herramienta",
        "no puedo responder con garantias",
        "sin herramientas no puedo",
    )
    if runs == [] and any(t in final_answer.strip().lower() for t in bad_templates):
        last_ai = find_last_assistant_real(state.get("messages", []) or [])
        raw = state.get("llm_raw_out") or (
            coerce_content_str(getattr(last_ai, "content", "")) if last_ai else ""
        )
        direct = state.get("llm_clean_out") or strip_think(raw)
        if direct:
            final_answer = direct
            try:
                summary["final_answer"] = direct
            except Exception:
                pass
            state["user_out"] = direct

    all_covered = True
    reasons: List[str] = []
    subqueries = analyzer.get("subqueries") or []
    logic_form = (analyzer.get("propositional_logic") or "").strip()
    active_skills_eff = resolve_effective_skills(state, skill_registry)
    invariant_violations: List[str] = compute_invariant_violations(
        subqueries=subqueries,
        logic_form=logic_form,
        planner_trajs=planner_trajs,
        executor_steps=executor_steps,
        runs_count=len(runs),
        input_object_count=input_object_count,
        active_skills_eff=active_skills_eff,
        planner_scope=planner_scope,
        is_placeholder_subquery=is_placeholder_subquery,
    )

    if not final_answer.strip():
        all_covered = False
        reasons.append("La respuesta final esta vacAa.")

    if "No se invoco ninguna herramienta" in summarizer_text and runs:
        all_covered = False
        reasons.append(
            "Inconsistencia: el SUMMARIZER dice que no hubo tools, "
            "pero tool_runs no esta vacio."
        )

    if invariant_violations:
        all_covered = False
        reasons.extend(invariant_violations)
        if has_coverage_partial(invariant_violations):
            warning = build_coverage_warning(
                input_object_count=input_object_count,
                runs_count=len(runs),
            )
            current_answer = final_answer if isinstance(final_answer, str) else ""
            if warning not in current_answer:
                final_answer = f"{warning}\n\n{current_answer}".strip()
                try:
                    summary["final_answer"] = final_answer
                except Exception:
                    pass

    if fail_fast and invariant_violations:
        fail_msg_lines = [
            "Ejecución detenida por fail-fast: se violaron invariantes estructurales.",
            "",
            f"Prompt: {user_prompt}",
            "Violaciones:",
        ]
        for idx, violation in enumerate(invariant_violations, start=1):
            fail_msg_lines.append(f"{idx}. {violation}")
        final_answer = "\n".join(fail_msg_lines)
        try:
            summary["final_answer"] = final_answer
        except Exception:
            pass

    if not reasons and all_covered:
        reasons.append("No se detectaron problemas obvios de cobertura.")

    validator = {
        "all_covered": all_covered,
        "reasoning": "\n".join(reasons),
    }

    validator_msg = AIMessage(
        content=(
            "### VALIDATOR\n\n"
            f"- all_covered: {all_covered}\n"
            f"- reasoning:\n{validator['reasoning']}"
        ),
        additional_kwargs={"pipeline_internal": True, "node": "validator"},
    )

    return {
        "validator": validator,
        "messages": [validator_msg],
        "pipeline_summary": summary,
        "summary": summary,
        "user_out": final_answer
        if isinstance(final_answer, str) and final_answer.strip()
        else state.get("user_out"),
    }
