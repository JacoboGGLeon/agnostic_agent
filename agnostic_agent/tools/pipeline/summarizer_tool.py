from __future__ import annotations

import html
import json
import re
from typing import Any, Callable, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def execute_summarizer_tool(
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
    build_user_answer_from_runs: Callable[[str, List[Dict[str, Any]], List[str] | None], str],
    is_technical_answer: Callable[[str], bool],
    find_last_assistant_real: Callable[[List[Any]], Any],
    extract_tool_calls: Callable[[Any], List[Dict[str, Any]]],
    coerce_content_str: Callable[[Any], str],
    strip_think: Callable[[str], str],
) -> Dict[str, Any]:
    messages = state["messages"]
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    last_user = user_messages[-1] if user_messages else None
    user_text = last_user.content if isinstance(last_user, HumanMessage) else ""
    user_prompt = state.get("user_prompt") or user_text
    runs = state.get("tool_runs", []) or []
    analyzer = state.get("analyzer") or {}

    def _pretty_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=json_default)
        except Exception:
            return repr(value)

    def _build_analyzer_text(an: Dict[str, Any]) -> str:
        if not an:
            return "Analyzer did not run or did not leave state."
        subqs = an.get("subqueries") or []
        logic_expr = an.get("propositional_logic") or "(not built)"
        payload = an.get("input_payload") or {}
        selection = state.get("_analyzer_skill_selection") or {}
        active_skills = resolve_effective_skills(state, skill_registry)
        lines = [
            f"Input payload: {_pretty_json(payload)}",
            f"Logica proposicional: {logic_expr}",
            f"Subconsultas ({len(subqs)}):",
            "Rol: ANALYZER elige skill.",
        ]
        for idx, sq in enumerate(subqs, start=1):
            lines.append(f"- q{idx} = {sq}")
        lines.append(f"Skills activas: {active_skills if active_skills else '[]'}")
        forced_skill_local = state.get("forced_skill")
        allowlist_local = state.get("skills_allowlist") or []
        if forced_skill_local and forced_skill_local != "Auto (Analyzer)":
            lines.append(f"Skill forzada (UI): {forced_skill_local}")
        if allowlist_local:
            lines.append(f"Skills allowlist (UI): {allowlist_local}")
        selected_skill = selection.get("selected_skill")
        selected_score = selection.get("score")
        selected_source = selection.get("source") or "unknown"
        if selected_skill:
            if selected_score is None:
                lines.append(
                    f"Skill seleccionada: {selected_skill} (score n/a, origen={selected_source})"
                )
            else:
                lines.append(
                    f"Skill seleccionada: {selected_skill} (score {selected_score}, origen={selected_source})"
                )
        return "\n".join(lines)

    def _build_planner_text() -> str:
        def _as_text_safe(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            try:
                return json.dumps(value, ensure_ascii=False, default=json_default)
            except Exception:
                return str(value)

        def _looks_like_js_object_repr(value: Any) -> bool:
            txt = _as_text_safe(value).strip().lower()
            if not txt:
                return False
            compact = re.sub(r"\s+", "", txt)
            if "[objectobject]" in compact or "objectobject" in compact:
                return True
            compact = compact.replace('"', "").replace("'", "")
            return "[objectobject]" in compact or "objectobject" in compact

        def _normalize_scope_values(values: Any) -> List[str]:
            if values is None:
                return []
            if not isinstance(values, list):
                values = [values]
            normalized: List[str] = []
            for item in values:
                txt = _as_text_safe(item).strip()
                if not txt:
                    continue
                if _looks_like_js_object_repr(txt):
                    continue
                normalized.append(txt)
            return normalized

        def _is_object_object_line(line: str) -> bool:
            return _looks_like_js_object_repr(line)

        planner_trajs = state.get("planner_trajs", []) or []
        if not planner_trajs:
            return "Planner did not build a tool plan."
        planner_scope = state.get("_planner_scope_internal") or {}
        out_lines: List[str] = []
        out_lines.append("Rol: PLANNER restringe tools+knowledge.")
        if planner_scope:
            scope_skills = _normalize_scope_values(planner_scope.get("active_skills", []))
            scope_tools = _normalize_scope_values(planner_scope.get("allowed_tools", []))
            scope_knowledge = _normalize_scope_values(planner_scope.get("allowed_knowledge", []))
            out_lines.append(
                f"Scope: skills={scope_skills}, tools={scope_tools}, knowledge={scope_knowledge}"
            )
            out_lines.append("")
        for i, tr in enumerate(planner_trajs, start=1):
            if isinstance(tr, dict):
                subq_val = tr.get("subquery", "")
                desc_val = tr.get("description", "")
            else:
                subq_val = getattr(tr, "subquery", "")
                desc_val = getattr(tr, "description", "")
                if not subq_val and not desc_val:
                    desc_val = _as_text_safe(tr)
            out_lines.append(f"Subquery {i}: {_as_text_safe(subq_val)}")
            out_lines.append("DAG:")
            raw_desc = _as_text_safe(desc_val).strip()
            if not raw_desc:
                out_lines.append("step 1: (empty)")
            else:
                for raw_line in raw_desc.splitlines():
                    line = raw_line.strip()
                    if _is_object_object_line(line):
                        continue
                    if line.lower().startswith("note:"):
                        out_lines.append(line)
                    elif line.startswith("step "):
                        out_lines.append(line)
                    else:
                        out_lines.append(f"step ?: {line}")
            if i < len(planner_trajs):
                out_lines.append("")
        analyzer_subqs = (state.get("analyzer") or {}).get("subqueries") or []
        exec_steps = state.get("executor_steps", []) or []
        if len(analyzer_subqs) != len(planner_trajs):
            out_lines.append("")
            out_lines.append(
                "[WARN] Cobertura Analyzer->Planner inconsistente: "
                f"subqueries={len(analyzer_subqs)} vs plans={len(planner_trajs)}"
            )
        if len(exec_steps) < len(analyzer_subqs):
            out_lines.append(
                "[WARN] Cobertura Analyzer->Executor parcial: "
                f"subqueries={len(analyzer_subqs)} vs calls={len(exec_steps)}"
            )
        joined = "\n".join(out_lines)
        for _ in range(3):
            joined = re.sub(r"(?i),?\s*['\"]?\[object\s*Object\]['\"]?\s*,?", "", joined)
            joined = re.sub(r"(?i)\[object\s*Object\]", "", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()

    def _build_executor_text() -> str:
        executor_steps = state.get("executor_steps", []) or []
        if not executor_steps:
            return "No tool execution happened for this query."
        lines: List[str] = [f"Se ejecutaron {len(executor_steps)} llamadas a herramientas:"]
        for idx, step in enumerate(executor_steps, start=1):
            lines.append(f"step {idx}:")
            lines.append(f"  tool_call_id: {step.get('tool_call_id')}")
            lines.append(f"  name: {step.get('tool_name')}")
            lines.append(f"  args: {_pretty_json(step.get('args', {}))}")
        return "\n".join(lines)

    def _build_catcher_text(tool_runs: List[Dict[str, Any]]) -> str:
        if not tool_runs:
            return "Catcher did not find tool results (tool_runs is empty)."
        lines: List[str] = [f"Catcher recopilo {len(tool_runs)} resultados de tools."]
        for idx, run in enumerate(tool_runs, start=1):
            lines.append(f"resultado {idx}:")
            lines.append(f"  tool: {run.get('name')}")
            lines.append(f"  args: {_pretty_json(run.get('args', {}))}")
            lines.append(f"  output_type: {type(run.get('output')).__name__}")
        return "\n".join(lines)

    def _normalize_text(s: str) -> str:
        if not isinstance(s, str):
            return s
        out = s
        for _ in range(3):
            decoded = html.unescape(out)
            if decoded == out:
                break
            out = decoded
        replacements = {
            "Ã¢â€ â€™": "->",
            "ÃƒÂ¡": "Ã¡",
            "ÃƒÂ©": "Ã©",
            "ÃƒÂ­": "Ã­",
            "ÃƒÂ³": "Ã³",
            "ÃƒÂº": "Ãº",
            "ÃƒÂ±": "Ã±",
            "ÃƒÂ": "Ã",
            "Ãƒâ€°": "Ã‰",
            "ÃƒÂ": "Ã",
            "Ãƒâ€œ": "Ã“",
            "ÃƒÅ¡": "Ãš",
            "Ãƒâ€˜": "Ã‘",
            "Ã°Å¸â€œÅ’": "[PIN]",
            "Ã°Å¸â€Â": "[SEARCH]",
            ",[object Object],": "",
            "[object Object]": "",
            ", [object Object],": "",
            "[objectObject]": "",
        }
        for bad, good in replacements.items():
            out = out.replace(bad, good)
        for _ in range(3):
            out = re.sub(r"(?i),?\s*['\"]?\[object\s*Object\]['\"]?\s*,?", "", out)
            out = re.sub(r"(?i)\[object\s*Object\]", "", out)
            out = out.replace("[object Object]", "")
        out = re.sub(r"(?im)^\s*step\s*\?:\s*$", "", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        out = re.sub(r",\s*,", ",", out)
        out = re.sub(r"\[\s*,", "[", out)
        out = re.sub(r",\s*\]", "]", out)
        return out.strip()

    def _build_pipeline_markdown(title: str, final_heading: str) -> str:
        def _fenced_block(text: str, lang: str = "text") -> str:
            body = text if isinstance(text, str) else _pretty_json(text)
            return f"````{lang}\n{body}\n````"

        return "\n\n".join(
            [
                f"**{title.upper()}**",
                "**ANALYZER**",
                _fenced_block(analyzer_text, "text"),
                "**PLANNER**",
                _fenced_block(planner_text, "text"),
                "**EXECUTOR**",
                _fenced_block(executor_text, "text"),
                "**CATCHER**",
                _fenced_block(catcher_text, "text"),
                "**SUMMARIZER (basado en herramientas)**",
                _fenced_block(summarizer_text, "text"),
                final_heading,
                user_answer,
            ]
        )

    def _build_deep_markdown() -> str:
        def _section(title: str, body: str) -> str:
            try:
                body_text = (
                    str(body)
                    if not isinstance(body, (dict, list))
                    else json.dumps(body, ensure_ascii=False, indent=2)
                )
            except Exception:
                body_text = "(Error serializando vista profunda)"
            body_text = body_text.replace("[object Object]", "")
            body_text = body_text.replace(",,", ",")
            return f"**{title}**\n```text\n{body_text.strip()}\n```"

        return "\n\n".join(
            [
                "**RESUMEN DEEP DEL PIPELINE**",
                _section("ANALYZER", analyzer_text),
                _section("PLANNER", planner_text),
                _section("EXECUTOR", executor_text),
                _section("CATCHER", catcher_text),
                _section("SUMMARIZER", summarizer_text),
                _section("RESPUESTA FINAL", user_answer),
            ]
        )

    active_skills = state.get("_active_skills_internal") or []
    if "capabilities_menu" in (active_skills or []):
        knowledge_selected = state.get("knowledge_selected") or []

        def _fmt_tool_sig(tool: Any) -> str:
            name = getattr(tool, "name", "tool")
            args = getattr(tool, "args", None)
            if isinstance(args, dict) and args:
                keys = ", ".join(args.keys())
                return f"{name}({keys})"
            return f"{name}()"

        lines: List[str] = []
        lines.append("## Menu de capacidades")
        lines.append("")
        lines.append(
            "Tu consulta no activo una skill especifica. Elige una skill y reintenta (o ajusta tu pregunta)."
        )
        lines.append("")
        lines.append("### Skills")
        if skill_registry:
            for skill in skill_registry.list_skills():
                tools_list = ", ".join(skill.tools) if skill.tools else "(sin tools)"
                kb_list = ", ".join(skill.knowledge) if skill.knowledge else "(sin knowledge)"
                lines.append(f"- `{skill.name}`: {skill.description}")
                lines.append(f"  - tools: {tools_list}")
                lines.append(f"  - knowledge: {kb_list}")
        else:
            lines.append("- (Skill registry no disponible)")
        lines.append("")
        lines.append("### Tools")
        if tools:
            for tool in tools:
                desc = getattr(tool, "description", "") or ""
                desc = desc.strip().splitlines()[0] if desc else ""
                lines.append(f"- `{_fmt_tool_sig(tool)}`" + (f": {desc}" if desc else ""))
        else:
            lines.append("- (No hay tools cargadas)")
        lines.append("")
        lines.append("### Knowledge")
        if knowledge_selected:
            for kb in knowledge_selected:
                if not isinstance(kb, dict):
                    continue
                nm = kb.get("name", "unknown")
                kind = kb.get("kind", "generic")
                desc = (kb.get("description") or "").strip()
                if desc:
                    lines.append(f"- `{nm}` ({kind}): {desc}")
                else:
                    lines.append(f"- `{nm}` ({kind})")
        else:
            lines.append("- (No hay knowledge activo para esta sesion)")
        lines.append("")
        lines.append("### Como elegir")
        lines.append("- Si quieres buscar en documentos/KB: usa `semantic_researcher`.")
        lines.append("- Si quieres calcular: usa `math_helper`.")
        lines.append("- Si quieres transformar texto: usa `text_basic`.")

        user_answer = "\n".join(lines)
        analyzer_text = "Skill de soporte `capabilities_menu` activada (menu de capacidades)."
        planner_text = "Planner no ejecuto tools: se devolvio menu determinista de capacidades."
        executor_text = "No se ejecuto ninguna herramienta."
        catcher_text = "No hubo tool runs."
        summarizer_text = "Respuesta generada sin LLM, basada en registros locales (skills/tools/knowledge)."
        user_answer = _normalize_text(user_answer)
        analyzer_text = _normalize_text(analyzer_text)
        planner_text = _normalize_text(planner_text)
        executor_text = _normalize_text(executor_text)
        catcher_text = _normalize_text(catcher_text)
        summarizer_text = _normalize_text(summarizer_text)

        summary_dict = {
            "analyzer": analyzer_text,
            "planner": planner_text,
            "executor": executor_text,
            "catcher": catcher_text,
            "summarizer": summarizer_text,
            "final_answer": user_answer,
        }
        answer_markdown = _build_pipeline_markdown(
            "Resumen del pipeline",
            "### RESPUESTA FINAL (modo usuario)",
        )
        final_ai = AIMessage(
            content=user_answer,
            additional_kwargs={"pipeline_internal": True, "node": "summarizer"},
        )
        return {
            "messages": [final_ai],
            "summary": summary_dict,
            "pipeline_summary": summary_dict,
            "dev_out": answer_markdown,
            "deep_out": _build_deep_markdown(),
            "user_out": user_answer,
        }

    if not runs:
        last_ai = find_last_assistant_real(messages)
        last_ai_has_tools = bool(extract_tool_calls(last_ai)) if last_ai else False
        llm_raw = state.get("llm_raw_out") or (
            coerce_content_str(getattr(last_ai, "content", "")) if last_ai else ""
        )
        llm_clean = state.get("llm_clean_out") or strip_think(llm_raw)

        is_dag_attempt = False
        if '"dag":' in llm_clean or "'dag':" in llm_clean:
            is_dag_attempt = True
        if is_dag_attempt:
            fallback_sys = (
                "Eres un asistente servicial y agnostico.\n"
                "El usuario te ha dicho algo que NO requiere herramientas externas, o bien "
                "las herramientas que intentaste usar no estan permitidas en el contexto actual.\n"
                "Responde de forma natural, util y amable en el idioma del usuario.\n"
                "NO inventes informacion."
            )
            try:
                base_chat_model = getattr(planner_llm, "bound", planner_llm)
                history_clean: List[Any] = []
                for message in messages:
                    if message == last_ai:
                        continue
                    if isinstance(message, HumanMessage):
                        history_clean.append(message)
                    elif isinstance(message, AIMessage):
                        if not getattr(message, "additional_kwargs", {}).get("pipeline_internal"):
                            history_clean.append(message)
                fallback_reply = base_chat_model.invoke(
                    [SystemMessage(content=fallback_sys)] + history_clean
                )
                fallback_content = getattr(fallback_reply, "content", "")
                user_answer = strip_think(fallback_content)
            except Exception:
                user_answer = f"Hola. He recibido tu mensaje: '{user_prompt}'"
        elif last_ai_has_tools:
            user_answer = (
                "Se planificaron llamadas a herramientas, pero no se obtuvo ninguna salida. "
                "Revisa EXECUTOR/CATCHER o el registro de tools."
            )
        else:
            if not llm_clean and llm_raw and llm_raw.strip():
                user_answer = (
                    "_(El modelo genero un razonamiento interno pero no una respuesta final. "
                    "Ver pestana 'Thinking' en el Inspector)_"
                )
            else:
                user_answer = llm_clean or "Que te gustaria hacer?"

        user_answer = _normalize_text(user_answer)
        _ = summarize_tool_runs(user_prompt, runs)
        analyzer_text = _build_analyzer_text(analyzer)
        planner_text = _build_planner_text()
        executor_text = _build_executor_text()
        catcher_text = _build_catcher_text(runs)
        summarizer_text = "No se invocaron herramientas. Respuesta directa del modelo (passthrough)."
        summary_dict = {
            "analyzer": analyzer_text,
            "planner": planner_text,
            "executor": executor_text,
            "catcher": catcher_text,
            "summarizer": summarizer_text,
            "final_answer": user_answer,
        }
    else:
        tools_summary_text = summarize_tool_runs(user_prompt, runs)
        analyzer_subqueries = analyzer.get("subqueries") if isinstance(analyzer, dict) else None
        try:
            deterministic_user_answer = build_user_answer_from_runs(
                user_prompt,
                runs,
                analyzer_subqueries if isinstance(analyzer_subqueries, list) else None,
            )
        except TypeError:
            deterministic_user_answer = build_user_answer_from_runs(user_prompt, runs)  # type: ignore[misc]
        user_answer = deterministic_user_answer

        hybrid_sys = (
            "You are an assistant that must answer only from verified tool evidence.\n"
            "Write a concise, user-facing answer in the same language as the user.\n"
            "Do not include internal traces, tool call IDs, steps, args, or debug logs.\n"
            "If evidence is insufficient, state that clearly without inventing facts."
        )
        if cfg and not cfg.enable_thinking:
            hybrid_sys += (
                "\n\nCRITICAL: DO NOT use <think> tags. Respond ONLY with the final natural language answer."
            )
        hybrid_user_msg = (
            f"User request:\n{user_prompt}\n\n"
            f"Verified tool evidence:\n{tools_summary_text}\n\n"
            f"Deterministic baseline answer:\n{deterministic_user_answer}\n\n"
            "Improve readability and clarity for the final user response."
        )

        try:
            hrm = planner_llm.invoke(
                [
                    SystemMessage(content=hybrid_sys),
                    HumanMessage(content=hybrid_user_msg),
                ]
            )
            llm_answer = strip_think(coerce_content_str(getattr(hrm, "content", ""))).strip()

            if llm_answer and not is_technical_answer(llm_answer):
                user_answer = llm_answer

            reasoning_from_final = ""
            if hasattr(hrm, "additional_kwargs") and isinstance(hrm.additional_kwargs, dict):
                reasoning_from_final = (
                    hrm.additional_kwargs.get("reasoning_content")
                    or hrm.additional_kwargs.get("reasoning")
                    or hrm.additional_kwargs.get("thoughts")
                    or ""
                )
            if isinstance(reasoning_from_final, str) and reasoning_from_final.strip():
                thinking_msg = AIMessage(
                    content="",
                    additional_kwargs={
                        "reasoning_content": reasoning_from_final.strip(),
                        "final_answer_thinking": True,
                    },
                )
                state.setdefault("messages", []).append(thinking_msg)
        except Exception:
            user_answer = deterministic_user_answer

        user_answer = strip_think(coerce_content_str(user_answer)).strip()
        if not user_answer or is_technical_answer(user_answer):
            user_answer = deterministic_user_answer

        analyzer_text = _build_analyzer_text(analyzer)
        planner_text = _build_planner_text()
        executor_text = _build_executor_text()
        catcher_text = _build_catcher_text(runs)
        summarizer_text = summarize_tool_runs_compact(runs)
        summary_dict = {
            "analyzer": analyzer_text,
            "planner": planner_text,
            "executor": executor_text,
            "catcher": catcher_text,
            "summarizer": summarizer_text,
            "final_answer": user_answer,
        }
    analyzer_text = _normalize_text(analyzer_text)
    planner_text = _normalize_text(planner_text)
    executor_text = _normalize_text(executor_text)
    catcher_text = _normalize_text(catcher_text)
    summarizer_text = _normalize_text(summarizer_text)
    user_answer = _normalize_text(user_answer)

    summary_dict["analyzer"] = analyzer_text
    summary_dict["planner"] = planner_text
    summary_dict["executor"] = executor_text
    summary_dict["catcher"] = catcher_text
    summary_dict["summarizer"] = summarizer_text
    summary_dict["final_answer"] = user_answer

    answer_markdown = _build_pipeline_markdown(
        "Resumen del pipeline",
        "**RESPUESTA FINAL (modo usuario)**",
    )
    user_out = strip_think(user_answer)
    final_ai = AIMessage(
        content=user_out,
        additional_kwargs={"pipeline_internal": True, "node": "summarizer"},
    )
    return {
        "messages": [final_ai],
        "summary": summary_dict,
        "pipeline_summary": summary_dict,
        "dev_out": answer_markdown,
        "deep_out": _build_deep_markdown(),
        "user_out": user_out,
    }


