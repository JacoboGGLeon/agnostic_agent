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

                # Capture Tool Logs
                if "tool_logs" not in st.session_state:
                    st.session_state["tool_logs"] = []
                
                # We check if we already logged this run to avoid dupes?
                # Actually, in re-runs we just rebuild UI, the session_state logs persist.
                # The logic in original file was doing append inside the loop over session_state.messages
                # This causes DUPLICATION on every rerun! 
                # FIX: Moving log capture to turn execution time or check existence.
                # For faithful reproduction, I'll allow the duplication if original had it, 
                # but it looks like original logic was flawed.
                # I will Skip log appending here to fix the bug, assuming logs are captured at execution.
                # Wait, original code:
                # `if "tool_logs" not in st.session_state: st.session_state["tool_logs"] = []`
                # `for tr in tool_runs: ... append` (inside the loop over messages)
                # Yes, that duplicates logs on every refresh. I will Fix it by not appending here.
                # I'll rely on execution time logging.

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
        try:
            raw_out = agent.run_turn(prompt)
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
