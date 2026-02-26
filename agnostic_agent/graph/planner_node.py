from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agnostic_agent.core.pipeline_v2.planner import (
    build_subqueries_from_prompt,
    flatten_tool_calls_by_subquery,
)

logger = logging.getLogger(__name__)


def _sanitize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"(?i),?\s*['\"]?\[object\s*object\]['\"]?\s*,?", "", text)
    text = text.replace("[object Object]", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_expected_id_constraints(subquery_text: str) -> Dict[str, str]:
    """
    Infer expected entity id constraints from subquery text in a generic way.
    Prefers explicit JSON fragments; falls back to generic ID-like tokens.
    """
    text = _sanitize_text(subquery_text)
    constraints: Dict[str, str] = {}
    if not text:
        return constraints

    for chunk in re.findall(r"\{[^{}]+\}", text):
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for key, value in obj.items():
            key_s = str(key)
            if key_s.endswith("_id") and value not in (None, ""):
                constraints[key_s] = str(value)

    if constraints:
        return constraints

    # Fallback: generic ID-like token (e.g., LOC-0010, ABC_1234, USER42)
    token_match = re.search(r"\b[A-Za-z]{2,}[A-Za-z0-9_-]*-[A-Za-z0-9_-]+\b", text)
    if token_match:
        constraints["entity_id"] = token_match.group(0)
    return constraints


def _collect_arg_id_fields(args: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(args, dict):
        return out
    for key, value in args.items():
        key_s = str(key)
        if key_s.endswith("_id") and value not in (None, ""):
            out[key_s] = str(value)
    return out


def _align_call_ids_to_subquery(
    *,
    subquery_text: str,
    tool_name: str,
    args: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Ensure call args entity IDs are semantically aligned with subquery IDs.
    Returns (ok, normalized_args, reason_if_blocked).
    """
    expected = _extract_expected_id_constraints(subquery_text)
    if not expected:
        return True, args, ""

    normalized = dict(args or {})
    actual = _collect_arg_id_fields(normalized)

    # No *_id in args but exactly one expected id -> inject it.
    if not actual and len(expected) == 1:
        exp_key, exp_val = next(iter(expected.items()))
        normalized.setdefault(exp_key, exp_val)
        return True, normalized, ""

    # If same key exists with wrong value, overwrite with expected deterministic value.
    changed = False
    for key, exp_val in expected.items():
        if key in normalized and str(normalized.get(key)) != str(exp_val):
            normalized[key] = exp_val
            changed = True

    if changed:
        return True, normalized, ""

    actual_after = _collect_arg_id_fields(normalized)
    if not actual_after:
        return True, normalized, ""

    # If all shared keys match, consider aligned.
    shared_keys = set(expected.keys()) & set(actual_after.keys())
    if shared_keys and all(expected[k] == actual_after[k] for k in shared_keys):
        return True, normalized, ""

    # If expected has one value and args has one *_id key, force alignment.
    if len(expected) == 1 and len(actual_after) == 1:
        _, exp_val = next(iter(expected.items()))
        act_key, act_val = next(iter(actual_after.items()))
        if act_val != exp_val:
            normalized[act_key] = exp_val
        return True, normalized, ""

    return (
        False,
        normalized,
        f"entity_mismatch tool={tool_name} expected={expected} actual={actual_after}",
    )


def format_rich_context(
    skills_reg: Any,
    tools_list: List[Any],
    knowledge_list: List[Dict[str, Any]],
    exclude_skills: Optional[List[str]] = None,
) -> str:
    lines = ["== CONTEXTO DEL SISTEMA (Capabilities) ==", ""]
    lines.append("### Y SKILLS (Estrategias Activas)")
    if skills_reg:
        all_skills = skills_reg.list_skills()
        if all_skills:
            for s in all_skills:
                if exclude_skills and s.name in exclude_skills:
                    continue
                knowledge_hint = ""
                if s.knowledge:
                    knowledge_hint = f" -> Opera sobre Knowledge: {s.knowledge}"
                tools_hint = ""
                if s.tools:
                    tools_hint = f" -> Orquesta Tools: {s.tools}"
                lines.append(f"@skill {{name={s.name}}}")
                lines.append(f"  Description: {s.description}")
                lines.append(f"  Expansion: {tools_hint}{knowledge_hint}")
                lines.append("")
        else:
            lines.append("(No skills loaded)")
    else:
        lines.append("(Skill Registry not available)")
    lines.append("")

    lines.append("### Y  TOOLS (Funciones ejecutables)")
    if tools_list:
        for t in tools_list:
            name = getattr(t, "name", "tool")
            desc = getattr(t, "description", str(t))
            metadata = getattr(t.func if hasattr(t, "func") else t, "_agnostic_metadata", None)
            if metadata:
                mode = metadata.get("mode", "public")
                input_schema = metadata.get("input_schema")
                output_schema = metadata.get("output_schema", {})
            else:
                mode = "public"
                output_schema = {}
                input_schema = None
                if hasattr(t, "args_schema") and t.args_schema:
                    try:
                        input_schema = t.args_schema.schema_json()
                    except Exception:
                        input_schema = str(t.args_schema)
            if not input_schema and hasattr(t, "args_schema") and t.args_schema:
                try:
                    input_schema = t.args_schema.schema_json()
                except Exception:
                    input_schema = str(t.args_schema)
            input_str = json.dumps(input_schema) if input_schema else "Any"
            output_str = json.dumps(output_schema) if output_schema else "{}"
            lines.append(f"@tool {{name={name}, mode={mode}}}")
            lines.append(f"  Description/WhatItDoes: {desc}")
            lines.append(f"  Input Schema: {input_str}")
            lines.append(f"  Output Schema: {output_str}")
            lines.append("")
    else:
        lines.append("(No tools available)")
    lines.append("")

    lines.append("### Ys KNOWLEDGE (Bases de Datos)")
    if knowledge_list:
        for knowledge in knowledge_list:
            knowledge_name = knowledge.get("name", "unknown")
            knowledge_desc = knowledge.get("description", "Sin descripcion")
            lines.append(f"@knowledge {{name={knowledge_name}}}")
            lines.append(f"  Description: {knowledge_desc}")
            lines.append("")
    else:
        lines.append("(No knowledge bases active)")

    return "\n".join(lines)


def _knowledge_name_or_path_matches(
    knowledge_obj: Dict[str, Any],
    required_knowledge_names: set[str],
) -> bool:
    if not isinstance(knowledge_obj, dict) or not required_knowledge_names:
        return False
    kb_name = str(knowledge_obj.get("name") or "").strip()
    if kb_name and kb_name in required_knowledge_names:
        return True
    cfg = knowledge_obj.get("config") if isinstance(knowledge_obj.get("config"), dict) else {}
    kb_path = str(cfg.get("path") or "").strip()
    if kb_path:
        base = os.path.basename(kb_path)
        if base in required_knowledge_names:
            return True
    return False


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
    msgs = state["messages"]
    knowledge_selected = state.get("knowledge_selected") or []
    analyzer = state.get("analyzer") or {}
    subqs = analyzer.get("subqueries") or []

    active_skills = resolve_effective_skills(state, skill_registry)
    skill_mode = len(active_skills) > 0
    required_tool_names = set()
    required_knowledge_names = set()
    if skill_mode and skill_registry:
        for skill_name in active_skills:
            skill = skill_registry.get_skill(skill_name)
            if skill:
                if skill.tools:
                    required_tool_names.update(skill.tools)
                if skill.knowledge:
                    required_knowledge_names.update(skill.knowledge)

    if skill_mode:
        if required_tool_names:
            active_tools = [t for t in tools if t.name in required_tool_names]
        else:
            active_tools = []
    else:
        active_tools = tools

    if skill_mode and required_knowledge_names and "*" not in required_knowledge_names:
        active_knowledge_objects = [
            knowledge
            for knowledge in knowledge_selected
            if _knowledge_name_or_path_matches(knowledge, required_knowledge_names)
        ]
    else:
        active_knowledge_objects = knowledge_selected

    planner_scope = {
        "skill_mode": skill_mode,
        "active_skills": list(active_skills),
        "allowed_tools": [t.name for t in active_tools],
        "allowed_knowledge": [
            k.get("name") for k in active_knowledge_objects if isinstance(k, dict)
        ],
    }

    rich_context_text = format_rich_context(
        skill_registry,
        active_tools,
        active_knowledge_objects,
        exclude_skills=active_skills if skill_mode else None,
    )

    from agnostic_agent.prompts import PLANNER_DAG_SYSTEM_PROMPT

    sys_content = PLANNER_DAG_SYSTEM_PROMPT
    if cfg and not cfg.enable_thinking:
        sys_content += "\n\nCRITICAL: DO NOT use <think> tags. Respond ONLY with the JSON DAG block."

    current_llm = planner_llm
    if skill_mode:
        base_model = getattr(planner_llm, "bound", planner_llm)
        current_llm = base_model.bind_tools(active_tools)
        logger.info("planner skill mode active; rebound tools=%s", len(active_tools))

    # Avoid cross-turn leakage: planner should use current subquery context only.
    history: List[Any] = []

    all_tool_calls: List[Dict[str, Any]] = []
    planner_blocks: List[Dict[str, Any]] = []
    plan_trajs: List[Any] = []
    global_llm_clean = ""
    global_llm_raw = ""

    if not subqs:
        user_messages = [m for m in msgs if isinstance(m, human_message_type)]
        last_user = user_messages[-1] if user_messages else None
        if isinstance(last_user, human_message_type):
            subqs = build_subqueries_from_prompt(last_user.content)
    if not subqs:
        subqs = build_subqueries_from_prompt(state.get("user_prompt") or "")

    seen_calls_keys = set()

    def _invoke_planner_subquery_with_retry(
        llm_obj: Any,
        sys_content_local: str,
        history_local: List[Any],
        user_msg_local: Any,
        retries: int,
    ) -> Any:
        last_exc: Optional[Exception] = None
        attempts = max(1, retries + 1)
        for _ in range(attempts):
            try:
                resp = llm_obj.invoke(
                    [system_message_type(content=sys_content_local)] + history_local[:-1] + [user_msg_local]
                )
                if isinstance(resp, ai_message_type) and isinstance(resp.content, str):
                    clean_content = resp.content
                    for _ in range(3):
                        clean_content = re.sub(
                            r"(?i),?\s*['\"]?\[object\s*Object\]['\"]?\s*,?",
                            "",
                            clean_content,
                        )
                        clean_content = clean_content.replace("[object Object]", "")
                    resp.content = clean_content
                content_str = str(getattr(resp, "content", ""))
                if "generations found in stream" in content_str.lower():
                    raise ValueError(f"Provider returned empty stream string: {content_str}")
                return resp
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Planner invocation failed without exception detail.")

    for i, subq in enumerate(subqs, start=1):
        try:
            logger.debug("planner subquery %s/%s: %s", i, len(subqs), subq)
            user_msg_content = f"""CONTEXTO DISPONIBLE:
{rich_context_text}

TAREA ACTUAL (Subquery {i} de {len(subqs)}):
Genera el DAG exclusivo para resolver: "{subq}"
"""
            user_msg = human_message_type(content=user_msg_content)
            response = _invoke_planner_subquery_with_retry(
                llm_obj=current_llm,
                sys_content_local=sys_content,
                history_local=history,
                user_msg_local=user_msg,
                retries=cfg.max_retries,
            )

            current_raw = _sanitize_text(response.content)
            global_llm_raw += f"\n\n--- Subquery {i}: {subq} ---\n{current_raw}"
            content_cleaned = strip_think(current_raw).strip()
            global_llm_clean += f"\n\n--- Plan {i}: {subq} ---\n{content_cleaned}"

            native_calls = normalize_toolcalls_list(getattr(response, "tool_calls", []))
            if not native_calls:
                native_calls = extract_tool_calls_from_jsonish_text(
                    coerce_content_str(getattr(response, "content", ""))
                )

            allowed_tool_names = {t.name for t in active_tools}
            subq_calls: List[Dict[str, Any]] = []
            desc_lines: List[str] = []

            for step_idx, nc in enumerate(native_calls, start=1):
                n_name_raw = nc.get("name")
                n_name = canonical_tool_name(n_name_raw)
                n_args = nc.get("args", {}) or {}
                n_id = nc.get("id") or f"step_native_{i}_{str(uuid.uuid4())[:4]}"
                if not n_name:
                    continue
                if skill_mode and n_name not in allowed_tool_names:
                    logger.warning(
                        "planner blocked tool outside skill scope; raw=%s canonical=%s",
                        n_name_raw,
                        n_name,
                    )
                    continue
                ok_align, aligned_args, blocked_reason = _align_call_ids_to_subquery(
                    subquery_text=str(subq),
                    tool_name=n_name,
                    args=n_args,
                )
                if not ok_align:
                    logger.warning("planner blocked mismatched entity call: %s", blocked_reason)
                    continue
                n_args = aligned_args
                try:
                    n_args_key = json.dumps(n_args, sort_keys=True)
                except Exception:
                    n_args_key = repr(n_args)
                dedup_key = (n_name, n_args_key)
                if dedup_key in seen_calls_keys:
                    logger.debug("planner duplicate tool call skipped: %s", n_name)
                    continue
                seen_calls_keys.add(dedup_key)
                call_obj = {
                    "name": n_name,
                    "args": n_args,
                    "id": n_id,
                    "type": "tool_call",
                }
                subq_calls.append(call_obj)
                all_tool_calls.append(call_obj)
                desc_lines.append(
                    f"step {step_idx}: id={n_id}, tool={n_name}, args={n_args_key}"
                )

            planner_blocks.append({"subquery_idx": i, "tool_calls": subq_calls})

            if not subq_calls:
                desc_lines.append(
                    "No native tool calls were generated for this subquery. (Text reasoning only)"
                )

            plan_trajs.append(
                planner_trajectory_type(
                    subquery=_sanitize_text(subq),
                    description=_sanitize_text("\n".join(desc_lines)),
                )
            )
        except Exception as exc:
            logger.exception("planner error in subquery %s", i)
            plan_trajs.append(
                planner_trajectory_type(
                    subquery=_sanitize_text(subq),
                    description=_sanitize_text(f"Error: {exc}"),
                )
            )

    normalized_trajs: List[Dict[str, str]] = []
    for tr in plan_trajs:
        if isinstance(tr, dict):
            sq = _sanitize_text(tr.get("subquery", ""))
            ds = _sanitize_text(tr.get("description", ""))
        else:
            sq = _sanitize_text(getattr(tr, "subquery", ""))
            ds = _sanitize_text(getattr(tr, "description", ""))
        normalized_trajs.append({"subquery": sq, "description": ds})

    normalized_calls = flatten_tool_calls_by_subquery(planner_blocks)
    calls_by_subquery: Dict[int, int] = {}
    for c in normalized_calls:
        calls_by_subquery[c.subquery_idx] = calls_by_subquery.get(c.subquery_idx, 0) + 1

    planner_calls_by_subquery: List[Dict[str, Any]] = []
    for idx, subq in enumerate(subqs, start=1):
        count = calls_by_subquery.get(idx, 0)
        skipped_reason = ""
        if count == 0 and idx - 1 < len(normalized_trajs):
            tr = normalized_trajs[idx - 1]
            desc = tr.get("description", "")
            if "No native tool calls were generated" in desc:
                skipped_reason = "no_tool_calls_generated"
        planner_calls_by_subquery.append(
            {
                "subquery_idx": idx,
                "subquery": _sanitize_text(subq),
                "planned_calls": count,
                "skipped_reason": skipped_reason,
            }
        )

    if normalized_calls:
        all_tool_calls = [
            {
                "name": c.name,
                "args": c.args,
                "id": f"call_s{c.subquery_idx}_{uuid.uuid4().hex[:10]}",
                "type": "tool_call",
            }
            for c in normalized_calls
        ]

    logger.info("planner total consolidated tool calls=%s", len(all_tool_calls))
    ai_msg = ai_message_type(
        content=_sanitize_text(global_llm_clean),
        tool_calls=all_tool_calls,
        additional_kwargs={"dag_raw": _sanitize_text(global_llm_raw)},
    )
    return {
        "messages": [ai_msg],
        "planner_trajs": normalized_trajs,
        "planner_calls_by_subquery": planner_calls_by_subquery,
        "llm_raw_out": _sanitize_text(global_llm_raw),
        "llm_clean_out": _sanitize_text(global_llm_clean),
        "_planner_scope_internal": planner_scope,
    }
