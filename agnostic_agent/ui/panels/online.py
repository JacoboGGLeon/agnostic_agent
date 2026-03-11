import datetime
import html
from pathlib import Path
from typing import Any, Dict, List

import markdown
import streamlit as st

from agnostic_agent.ui.panels.helpers import (
    as_text,
    extract_tool_runs,
    get_raw_state,
    next_id,
    normalize_output,
    strip_user_prefix,
    render_markdown,
)
from agnostic_agent.ui.panels.inspector import render_inspector


def _format_skill_option(skill_name: str) -> str:
    if skill_name == "Auto (Analyzer)":
        return "Auto"
    return str(skill_name).replace("_", " ").title()


def _display_name_from_path(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    name = Path(value).name or value
    stem = Path(name).stem or name
    return stem.replace("_", " ")


def _world_summary_payload(skill_obj) -> Dict[str, Any]:
    if skill_obj is None:
        return {
            "world_label": "Auto",
            "world_description": "Auto: el Agentic OS selecciona el mundo adecuado para cada solicitud.",
            "tools": [],
            "knowledge": [],
        }
    ui_meta = getattr(skill_obj, "ui", {}) if isinstance(getattr(skill_obj, "ui", {}), dict) else {}
    world_label = str(ui_meta.get("world_label") or skill_obj.name).strip()
    world_description = str(ui_meta.get("world_description") or skill_obj.description or "").strip()
    tools = [str(tool_name).strip() for tool_name in (getattr(skill_obj, "tools", []) or []) if str(tool_name).strip()]
    knowledge_all = [
        _display_name_from_path(source)
        for source in (getattr(skill_obj, "knowledge", []) or [])
        if _display_name_from_path(source)
    ]
    return {
        "world_label": world_label,
        "world_description": world_description,
        "tools": tools,
        "knowledge": knowledge_all,
    }


def _render_mode_selector(skills: List[str]) -> str:
    st.markdown(
        """
        <div class="composer-toolbar-shell">
          <div class="composer-toolbar-title">Modo de trabajo</div>
          <div class="composer-toolbar-copy">Fija un mundo si quieres trabajar con un universo concreto. Si no, deja Auto.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.selectbox(
        "Modo de trabajo",
        ["Auto (Analyzer)"] + skills,
        index=0,
        key="debug_skill_selector",
        format_func=_format_skill_option,
        label_visibility="collapsed",
        help="Selecciona un mundo especifico para fijar el contexto de trabajo.",
    )


def _render_active_mode_summary(selected_skill: str, skill_obj: Any) -> None:
    payload = _world_summary_payload(skill_obj if selected_skill != "Auto (Analyzer)" else None)
    world_label = str(payload.get("world_label") or "Auto").strip()
    world_description = str(payload.get("world_description") or "").strip()
    tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
    knowledge = payload.get("knowledge") if isinstance(payload.get("knowledge"), list) else []
    tool_preview = tools[:4]
    knowledge_preview = knowledge[:3]
    tools_markup = "".join(f'<span class="composer-pill">{html.escape(str(tool))}</span>' for tool in tool_preview)
    knowledge_markup = "".join(
        f'<span class="composer-pill composer-pill-subtle">{html.escape(str(item))}</span>'
        for item in knowledge_preview
    )
    tool_suffix = (
        f'<span class="composer-caption">+{len(tools) - len(tool_preview)} tools</span>'
        if len(tools) > len(tool_preview)
        else ""
    )
    knowledge_suffix = (
        f'<span class="composer-caption">+{len(knowledge) - len(knowledge_preview)} fuentes</span>'
        if len(knowledge) > len(knowledge_preview)
        else ""
    )
    mode_label = "Auto" if selected_skill == "Auto (Analyzer)" else _format_skill_option(selected_skill)
    st.markdown(
        f"""
        <div class="composer-context-card">
          <div class="composer-context-head">
            <div>
              <div class="composer-fixed-kicker">Modo activo</div>
              <div class="composer-context-title">{html.escape(world_label)}</div>
              <div class="composer-context-copy">{html.escape(world_description)}</div>
            </div>
            <div class="composer-fixed-mode">{html.escape(mode_label)}</div>
          </div>
          <div class="composer-fixed-meta">
            <div class="composer-fixed-group">
              <div class="composer-section-label">Tools</div>
              <div class="composer-pill-row">{tools_markup or '<span class="composer-caption">Orquestacion automatica</span>'}{tool_suffix}</div>
            </div>
            <div class="composer-fixed-group">
              <div class="composer-section-label">Knowledge</div>
              <div class="composer-pill-row">{knowledge_markup or '<span class="composer-caption">Contexto automatico</span>'}{knowledge_suffix}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_online_tab(agent_factory, *, show_history: bool = True, show_inspector: bool = True):
    has_interaction = bool(st.session_state.get("messages"))
    agent_error = str(st.session_state.get("agent_init_error") or "").strip()

    if agent_error:
        st.error(f"Error iniciando agente: {agent_error}")

    skills: List[str] = []
    agent = agent_factory()
    if agent and agent.skill_registry:
        skills = [s.name for s in agent.skill_registry.list_skills()]

    selected_skill = _render_mode_selector(skills)
    active_skill_obj = None
    if selected_skill != "Auto (Analyzer)" and agent and agent.skill_registry:
        active_skill_obj = agent.skill_registry.get_skill(selected_skill)
    _render_active_mode_summary(selected_skill, active_skill_obj)

    if show_history and has_interaction:
        feed_col, insp_col = st.columns([2.2, 1.0], gap="large")

        with feed_col:
            with st.container(border=True):
                st.markdown('<div class="studio-feed-marker"></div>', unsafe_allow_html=True)
                for msg in st.session_state.messages:
                    role = msg.get("role", "user")

                    if role == "user":
                        with st.chat_message("user"):
                            content = msg.get("content", "")
                            st.markdown(
                                f"""
                                <div class="bubble-user">
                                  {html.escape(content)}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    elif role == "assistant":
                        out = msg.get("out") or {}
                        content = msg.get("content") or ""
                        used_mode = out.get("agent_mode", "")
                        raw_state = get_raw_state(out)
                        tool_runs = extract_tool_runs(out, raw_state)

                        badge_html = (
                            f'<span class="badge" style="font-size:10px; padding:2px 6px; margin-left:8px; opacity:0.7;">{used_mode}</span>'
                            if used_mode
                            else ""
                        )

                        with st.chat_message("assistant"):
                            raw_html = render_markdown(content or "_(sin respuesta)_")

                            st.markdown(
                                f"""
                                <div class="bubble-agent">
                                  <div style="font-size: 0.8em; opacity: 0.85; margin-bottom: 4px; display:flex; justify-content:flex-end; align-items:center; gap:8px;">
                                    <span>Respuesta {badge_html}</span>
                                    <span style="font-size:1rem;">&#129302;</span>
                                  </div>
                                  <div class="bubble-content">{raw_html}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            meta_col, inspect_col = st.columns([1.8, 1.0])
                            with meta_col:
                                st.caption(f"id={msg.get('id')} - tools:{len(tool_runs)}")
                            with inspect_col:
                                if st.button(
                                    "Inspect",
                                    key=f"inspect_{msg.get('id')}",
                                    use_container_width=True,
                                ):
                                    st.session_state.selected_msg_id = msg.get("id")
                                    st.toast(f"Inspector -> id={msg.get('id')}")
                                    st.rerun()

        with insp_col:
            if show_inspector:
                render_inspector()

    st.markdown('<div class="studio-feed-spacer"></div>', unsafe_allow_html=True)
    prompt = st.chat_input("Escribe tu mensaje...")

    if prompt:
        uid = next_id()
        st.session_state.messages.append(
            {"id": uid, "role": "user", "content": prompt, "out": {"agent_mode": "unified"}}
        )

        run_metadata = {}
        run_metadata["conversation_history_enabled"] = bool(
            st.session_state.get("conversation_history_enabled", True)
        )
        if selected_skill and selected_skill != "Auto (Analyzer)":
            # New path: explicit allowlist of active skills.
            run_metadata["skills_allowlist"] = [selected_skill]
            # Legacy compatibility path.
            run_metadata["forced_skill"] = selected_skill

        try:
            raw_out = agent.run_turn({"user_prompt": prompt, "metadata": run_metadata})
        except Exception as e:
            st.error(f"Error corriendo agente: {e}")
            st.stop()

        out = normalize_output(raw_out)
        aid = next_id()
        st.session_state.messages.append(
            {
                "id": aid,
                "role": "assistant",
                "content": strip_user_prefix(as_text(out.get("user_out"))),
                "out": out,
            }
        )

        raw_state_run = get_raw_state(out)
        tool_runs_run = extract_tool_runs(out, raw_state_run)

        if "tool_logs" not in st.session_state:
            st.session_state["tool_logs"] = []

        for tr in tool_runs_run:
            st.session_state["tool_logs"].append(
                {
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "tool": tr.get("name", "unknown"),
                    "input": tr.get("args", {}),
                    "output": tr.get("output", ""),
                }
            )

        st.session_state.selected_msg_id = aid
        st.rerun()
