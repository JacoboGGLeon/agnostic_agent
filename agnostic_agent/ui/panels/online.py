import streamlit as st
import html
import markdown
import datetime
from agnostic_agent.ui.panels.helpers import (
    get_raw_state, extract_tool_runs, next_id, normalize_output, 
    strip_user_prefix, as_text
)
from agnostic_agent.ui.panels.inspector import render_inspector

def render_online_tab(agent_factory):
    # -------- MAIN SCROLLABLE AREA (Chat + Inspector) --------
    # Adjust height as needed. 650px allows space for the fixed footer on standard screens.
    # -------- MAIN SCROLLABLE AREA (Chat + Inspector) --------
    # Adjust height as needed. 650px allows space for the fixed footer on standard screens.
    # Using a fixed height can sometimes cause layout "jumps" or double scrollbars.
    # We will try a slightly different approach: just columns, relying on main page scroll for now
    # if the container was causing the hang. But user explicitly asked for "partir en 2".
    # Let's keep the container but ensure it doesn't break the event loop.
    
    with st.container(height=600, border=False):
        feed_col, insp_col = st.columns([2.2, 1.0], gap="large")

        # -------- FEED (left) --------
        with feed_col:
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
                            unsafe_allow_html=True
                        )

                elif role == "assistant":
                    out = msg.get("out") or {}
                    content = msg.get("content") or ""
                    used_mode = out.get("agent_mode", "")
                    
                    raw_state = get_raw_state(out)
                    tool_runs = extract_tool_runs(out, raw_state)

                    # Badge
                    badge_html = f'<span class="badge" style="font-size:10px; padding:2px 6px; margin-left:8px; opacity:0.7;">{used_mode}</span>' if used_mode else ""
                    
                    with st.chat_message("assistant"):
                        try:
                            raw_html = markdown.markdown(content or "_(sin respuesta)_", extensions=['extra'])
                        except:
                            raw_html = html.escape(content or "_(sin respuesta)_").replace("\n", "<br>")

                        st.markdown(
                            f"""
                            <div class="bubble-agent">
                              <div style="font-size: 0.8em; opacity: 0.8; margin-bottom: 4px;">👤 Respuesta {badge_html} <span class="hint">id={msg.get('id')}</span></div>
                              <div class="bubble-content">{raw_html}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        c1, c2, c3 = st.columns([1.2, 1.0, 0.8])
                        with c1:
                            st.caption(f"🛠 tools: {len(tool_runs)}")
                        with c3:
                            if st.button("🔎 Inspect", key=f"inspect_{msg.get('id')}", use_container_width=True):
                                st.session_state.selected_msg_id = msg.get("id")
                                st.toast(f"Inspector → id={msg.get('id')}", icon="🔎")
                                st.rerun()

        # -------- INSPECTOR (right) --------
        with insp_col:
            render_inspector()

    # -------- SKILL SELECTOR (Fixed Layout - Bottom) --------
    # This sits outside the scrollable container, effectively creating a footer
    st.markdown("---") # Visual separator
    
    # Helper to get skills
    skills: List[str] = []
    agent = agent_factory() # Instantiate temp agent to get registry
    if agent and agent.skill_registry:
         skills = [s.name for s in agent.skill_registry.list_skills()]
    
    # Layout for Skill Selector relative to Input
    # We use columns to constrain width if desired, or full width
    sk_col, _ = st.columns([1, 1])
    with sk_col:
        selected_skill = st.selectbox(
            "Skill de Prueba (Forzar contexto)", 
            ["Auto (Analyzer)"] + skills,
            index=0,
            key="debug_skill_selector",
            help="Selecciona una skill específica para ver sus herramientas asociadas."
        )
    
        # Display Active Tools for Selected Skill
        if selected_skill != "Auto (Analyzer)" and agent.skill_registry:
            skill_obj = agent.skill_registry.get_skill(selected_skill)
            if skill_obj:
                tools_str = " | ".join([f"`{t}`" for t in skill_obj.tools])
                st.caption(f"🔧 **Tools Activas**: {tools_str}")
                if skill_obj.knowledge:
                     know_str = ", ".join(skill_obj.knowledge)
                     st.caption(f"📚 **Knowledge**: {know_str}")


    # -------- INPUT --------
    prompt = st.chat_input("Escribe tu mensaje…")

    if prompt:
        uid = next_id()
        msg_payload = {
            "id": uid, 
            "role": "user", 
            "content": prompt, 
            "out": {"agent_mode": "unified"} 
        }
        st.session_state.messages.append(msg_payload)

        agent = agent_factory()
        
        # Build metadata with forced skill if selected
        run_metadata = {}
        if selected_skill and selected_skill != "Auto (Analyzer)":
            run_metadata["forced_skill"] = selected_skill

        try:
            # Pass metadata to run_turn
            raw_out = agent.run_turn({
                "user_prompt": prompt,
                "metadata": run_metadata
            })
        except Exception as e:
            st.error(f"Error corriendo agente: {e}")
            st.stop()

        out = normalize_output(raw_out)
        aid = next_id()
        st.session_state.messages.append(
            {"id": aid, "role": "assistant", "content": strip_user_prefix(as_text(out.get("user_out"))), "out": out}
        )
        
        # Log tools HERE to avoid duplication on re-run
        raw_state_run = get_raw_state(out)
        tool_runs_run = extract_tool_runs(out, raw_state_run)
        
        if "tool_logs" not in st.session_state:
            st.session_state["tool_logs"] = []
            
        for tr in tool_runs_run:
            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "tool": tr.get("name", "unknown"),
                "input": tr.get("args", {}),
                "output": tr.get("output", "")
            }
            st.session_state["tool_logs"].append(log_entry)

        st.session_state.selected_msg_id = aid
        st.rerun()
