import streamlit as st
import json
import os
from agnostic_agent.ui.panels.helpers import next_id

def render_sidebar():
    with st.sidebar:
        # Header
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/BBVA_2019.svg/1280px-BBVA_2019.svg.png",
            width=140,
        )
        st.markdown("### Agentic Lab · Settings")

        # Inspector Toggle
        st.markdown("#### 🧭 Inspector")
        show_inspector = st.toggle("Activar Inspector", value=True, key="show_inspector")
        
        if show_inspector:
            st.caption("Vistas:")
            st.checkbox("🧠 Thinking", value=True, key="show_thinking_tab")
            st.checkbox("🧠 Deep", value=True, key="show_deep_tab")
            st.checkbox("🔍 Dev", value=True, key="show_dev_tab")
        
        st.divider()

        # Session / Transcript
        st.caption(f"Mensajes: {len(st.session_state.messages)}")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Limpiar", use_container_width=True):
                st.session_state.messages = []
                st.session_state.agent = None
                st.session_state.selected_msg_id = None
                st.toast("Chat reiniciado.", icon="🧹")
                st.rerun()
                
        with c2:
            if st.button("⬇️ Export", use_container_width=True):
                export = {"messages": st.session_state.messages}
                st.session_state.export_json = json.dumps(export, ensure_ascii=False, indent=2)
                st.toast("Transcript listo.", icon="⬇️")

        if isinstance(st.session_state.export_json, str):
            st.download_button(
                "Descargar JSON",
                data=st.session_state.export_json,
                file_name="transcript.json",
                mime="application/json",
                use_container_width=True,
            )

        st.divider()

        # Models Section
        st.markdown("#### 🤖 Models")
        
        llm_name = os.getenv("LLM_SERVED_NAME", "custom-llm-model")
        emb_name = os.getenv("EMB_SERVED_NAME", "custom-embedding-model")
        
        st.text_input("Model Name", value=llm_name, disabled=True, key="planner_model_name_display")
        st.caption("Embedding Server")
        st.text_input("Emb Name", value=emb_name, disabled=True, key="emb_model_name_display")
