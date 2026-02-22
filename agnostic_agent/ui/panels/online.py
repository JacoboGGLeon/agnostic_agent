import datetime
import html
from typing import List

try:
    import markdown
except Exception:
    markdown = None
import streamlit as st

from agnostic_agent.ui.panels.helpers import (
    as_text,
    extract_tool_runs,
    get_raw_state,
    next_id,
    normalize_output,
    strip_user_prefix,
)
from agnostic_agent.ui.panels.inspector import render_inspector


def render_online_tab(agent_factory):
    # Row 1: [Online Chat box][Inspector right sidebar-like panel]
    if "inspector_right_open" not in st.session_state:
        st.session_state.inspector_right_open = True

    # Trigger below top bar: open/close right inspector
    _, trigger_col_r = st.columns([0.96, 0.04], gap="small")
    with trigger_col_r:
        if st.button("Inspector", key="toggle_inspector_right_icon", use_container_width=True):
            st.session_state.inspector_right_open = not st.session_state.inspector_right_open
            st.rerun()

    if st.session_state.inspector_right_open:
        feed_col, insp_col = st.columns([2.05, 0.95], gap="large")
    else:
        feed_col = st.container()
        insp_col = None

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
                        try:
                            if markdown is not None:
                                raw_html = markdown.markdown(
                                    content or "_(sin respuesta)_", extensions=["extra"]
                                )
                            else:
                                raise RuntimeError("markdown package is not available")
                        except Exception:
                            raw_html = html.escape(content or "_(sin respuesta)_").replace(
                                "\n", "<br>"
                            )

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
    if insp_col is not None:
        with insp_col:
            st.markdown('<div id="right-inspector-anchor"></div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="right-inspector-head">
                  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/BBVA_2019.svg/1280px-BBVA_2019.svg.png" alt="BBVA"/>
                  <div class="right-inspector-title">Agentic Lab - Inspector</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="right-inspector-divider"></div>', unsafe_allow_html=True)
            render_inspector(show_title=False, boxed=False)

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

