import json
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from agnostic_agent.ui.panels.helpers import (
    assistant_messages,
    as_text,
    card_code,
    extract_summary_deep,
    extract_thinking,
    extract_tool_runs,
    find_message_by_id,
    get_raw_state,
    render_markdown,
    render_tool_runs,
    strip_user_prefix,
)


def _get_summary_v2(raw_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_state, dict):
        return None
    pipeline_v2 = raw_state.get("pipeline_v2")
    if not isinstance(pipeline_v2, dict):
        return None
    deep_v2 = pipeline_v2.get("deep_out")
    if not isinstance(deep_v2, dict):
        return None

    summary_v2 = deep_v2.get("summary")
    if isinstance(summary_v2, dict):
        return summary_v2

    artifacts = deep_v2.get("artifacts")
    if isinstance(artifacts, dict):
        fallback = artifacts.get("summary_v2")
        if isinstance(fallback, dict):
            return fallback
    return None


def _render_section_kv(title: str, section: Dict[str, Any], skip_keys: Optional[set] = None) -> None:
    skip_keys = skip_keys or set()
    rows: List[Dict[str, str]] = []
    for k, v in section.items():
        if k in skip_keys:
            continue
        if isinstance(v, (dict, list)):
            value = json.dumps(v, ensure_ascii=False)
        else:
            value = str(v)
        rows.append({"field": str(k), "value": value})

    if rows:
        st.markdown(f"###### {title}")
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _extract_planner_trajs(raw_state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(raw_state, dict):
        return []
    candidates: List[Any] = []
    candidates.append(raw_state.get("planner_trajs"))
    state_obj = raw_state.get("state")
    if isinstance(state_obj, dict):
        candidates.append(state_obj.get("planner_trajs"))
    pipeline_v2 = raw_state.get("pipeline_v2")
    if isinstance(pipeline_v2, dict):
        deep_out = pipeline_v2.get("deep_out")
        if isinstance(deep_out, dict):
            raw = deep_out.get("raw")
            if isinstance(raw, dict):
                candidates.append(raw.get("planner_trajs"))

    for trajs in candidates:
        if isinstance(trajs, list) and trajs:
            normalized: List[Dict[str, Any]] = []
            for item in trajs:
                if isinstance(item, dict):
                    normalized.append(item)
            if normalized:
                return normalized
    return []


def _parse_planner_description(desc: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not isinstance(desc, str) or not desc.strip():
        return rows

    for raw_line in desc.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^step\s+(\d+)\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if not m:
            continue
        step = m.group(1)
        tail = m.group(2).strip()
        call_id = ""
        tool = ""
        args = ""

        id_match = re.search(r"id=([^,\s]+)", tail)
        if id_match:
            call_id = id_match.group(1).strip()
        tool_match = re.search(r"tool=([^,\s]+)", tail)
        if tool_match:
            tool = tool_match.group(1).strip()
        args_match = re.search(r"args=(\{.*\})$", tail)
        if args_match:
            args = args_match.group(1).strip()

        rows.append(
            {
                "step": step,
                "tool_call_id": call_id,
                "tool": tool,
                "args": args,
            }
        )
    return rows


def _render_planner_dag(raw_state: Optional[Dict[str, Any]]) -> None:
    trajs = _extract_planner_trajs(raw_state)
    if not trajs:
        return

    st.markdown("###### Planner DAG (step x step)")
    for i, traj in enumerate(trajs, start=1):
        subquery = str(traj.get("subquery", "")).strip()
        desc = str(traj.get("description", "")).strip()
        label = f"Subquery {i}"
        if subquery:
            label = f"{label}: {subquery}"
        with st.expander(label, expanded=False):
            rows = _parse_planner_description(desc)
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.markdown("_(sin pasos parseables)_")
                if desc:
                    st.code(desc, language="text")


def render_inspector():
    if not st.session_state.get("show_inspector", True):
        st.info("Inspector oculto. Activalo en la barra lateral.")
        return

    with st.container(border=True):
        st.markdown("### Inspector")

        a_msgs = assistant_messages()
        if not a_msgs:
            st.info("Aun no hay respuestas del agente. Escribe algo para empezar.")
            return

        ids = [m["id"] for m in a_msgs]

        def label(mid: int) -> str:
            m = find_message_by_id(mid) or {}
            out = m.get("out") or {}
            text = strip_user_prefix(as_text(out.get("user_out"))).replace("\n", " ").strip()
            text = (text[:60] + "...") if len(text) > 60 else text
            return f"id={mid} - {text or '(sin texto)'}"

        if st.session_state.selected_msg_id not in ids:
            st.session_state.selected_msg_id = ids[-1]

        idx = 0
        try:
            idx = ids.index(st.session_state.selected_msg_id)
        except ValueError:
            pass

        sel = st.selectbox(
            "Mensaje seleccionado",
            options=ids,
            index=idx,
            format_func=label,
            key="inspector_selectbox",
        )
        st.session_state.selected_msg_id = sel

        m = find_message_by_id(st.session_state.selected_msg_id) or {}
        out = m.get("out") or {}
        raw_state = get_raw_state(out)

        thinking = extract_thinking(raw_state)
        deep_txt = extract_summary_deep(raw_state, as_text(out.get("deep_out")))
        tool_runs = extract_tool_runs(out, raw_state)

        tab_specs: List[Tuple[str, str]] = []
        if st.session_state.get("show_thinking_tab", True):
            tab_specs.append(("Thinking", "thinking"))
        if st.session_state.get("show_deep_tab", True):
            tab_specs.append(("Deep", "deep"))
        if st.session_state.get("show_dev_tab", True):
            tab_specs.append(("Dev", "dev"))

        if not tab_specs:
            return

        tabs = st.tabs([t[0] for t in tab_specs])
        for (_, tab_key), tab in zip(tab_specs, tabs):
            with tab:
                if tab_key == "thinking":
                    card_code(
                        "Pensamiento (thinking)",
                        thinking,
                        icon="Thinking",
                        hint="reasoning_content",
                    )
                elif tab_key == "deep":
                    content_to_show = deep_txt if deep_txt else "_(vacio / sin resumen)_"
                    st.markdown("##### Vista profunda (deep_out / summary)")

                    summary_v2 = _get_summary_v2(raw_state)
                    if isinstance(summary_v2, dict) and summary_v2:
                        section_order = [
                            ("Analyzer", "analyzer"),
                            ("Planner", "planner"),
                            ("Executor", "executor"),
                            ("Catcher", "catcher"),
                            ("Summarizer", "summarizer"),
                            ("Validator", "validator"),
                            ("Metrics", "metrics"),
                            ("Tool Outputs", "tool_outputs"),
                        ]
                        for title, key in section_order:
                            section = summary_v2.get(key)
                            if not isinstance(section, dict) or not section:
                                continue
                            if key == "analyzer":
                                _render_section_kv(title, section, skip_keys={"subquery_rows"})
                                rows = section.get("subquery_rows")
                                if isinstance(rows, list) and rows:
                                    st.markdown("###### Analyzer Subqueries")
                                    st.dataframe(rows, use_container_width=True, hide_index=True)
                            elif key == "planner":
                                _render_section_kv(title, section, skip_keys={"planner_call_rows"})
                                rows = section.get("planner_call_rows")
                                if isinstance(rows, list) and rows:
                                    st.markdown("###### Planner Calls")
                                    st.dataframe(rows, use_container_width=True, hide_index=True)
                                _render_planner_dag(raw_state)
                            elif key == "validator":
                                _render_section_kv(title, section, skip_keys={"coverage_report"})
                                coverage = section.get("coverage_report")
                                if isinstance(coverage, list) and coverage:
                                    st.markdown("###### Coverage Report")
                                    st.dataframe(coverage, use_container_width=True, hide_index=True)
                            elif key == "tool_outputs":
                                runs = section.get("runs")
                                if isinstance(runs, list) and runs:
                                    st.markdown("###### Tool Outputs")
                                    with st.expander("Ver deep Tool Outputs raw", expanded=False):
                                        st.code(json.dumps(runs, ensure_ascii=False, indent=2), language="json")
                            else:
                                _render_section_kv(title, section)
                    else:
                        st.markdown(render_markdown(content_to_show), unsafe_allow_html=True)

                    with st.expander("Ver deep markdown raw", expanded=False):
                        st.code(content_to_show, language="markdown")
                elif tab_key == "dev":
                    render_tool_runs(tool_runs)
                    with st.expander("raw_state (debug)", expanded=False):
                        if isinstance(raw_state, dict) and raw_state:
                            st.json(raw_state)
                        else:
                            st.markdown("_(sin raw_state)_")
