import datetime
import html
from typing import List

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


def render_online_tab(agent_factory):
    # Row 1: [Online Chat box][Inspector box]
    feed_col, insp_col = st.columns([2.2, 1.0], gap="large")

    with feed_col:
        with st.container(border=True):
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
        render_inspector()

    # Row 2: [Skill selector + input]
    st.markdown("---")

    skills: List[str] = []
    agent = agent_factory()
    if agent and agent.skill_registry:
        skills = [s.name for s in agent.skill_registry.list_skills()]

    selected_skill = st.selectbox(
        "Skill de Prueba (Forzar contexto)",
        ["Auto (Analyzer)"] + skills,
        index=0,
        key="debug_skill_selector",
        help="Selecciona una skill especifica para ver sus herramientas asociadas.",
    )

    if selected_skill != "Auto (Analyzer)" and agent and agent.skill_registry:
        skill_obj = agent.skill_registry.get_skill(selected_skill)
        if skill_obj:
            tools_str = " | ".join([f"`{tool_name}`" for tool_name in skill_obj.tools])
            st.caption(f"Tools Activas: {tools_str}")
            if skill_obj.knowledge:
                know_str = ", ".join(skill_obj.knowledge)
                st.caption(f"Knowledge: {know_str}")

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
