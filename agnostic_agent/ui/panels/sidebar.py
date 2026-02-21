import json
import os

import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/BBVA_2019.svg/1280px-BBVA_2019.svg.png",
            width=140,
        )
        st.markdown("### Agentic Lab · Settings")

        st.radio(
            "Theme",
            options=["dark", "light"],
            index=0 if st.session_state.get("theme_mode", "dark") == "dark" else 1,
            key="theme_mode",
            horizontal=True,
        )

        st.markdown("#### Inspector")
        show_inspector = st.toggle("Activar Inspector", value=True, key="show_inspector")

        if show_inspector:
            st.caption("Vistas:")
            st.checkbox("Thinking", value=True, key="show_thinking_tab")
            st.checkbox("Deep", value=True, key="show_deep_tab")
            st.checkbox("Dev", value=True, key="show_dev_tab")

        st.divider()

        st.caption(f"Mensajes: {len(st.session_state.messages)}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Limpiar", use_container_width=True):
                st.session_state.messages = []
                st.session_state.agent = None
                st.session_state.selected_msg_id = None
                st.toast("Chat reiniciado.")
                st.rerun()

        with c2:
            if st.button("Export", use_container_width=True):
                export = {"messages": st.session_state.messages}
                st.session_state.export_json = json.dumps(export, ensure_ascii=False, indent=2)
                st.toast("Transcript listo.")

        if isinstance(st.session_state.export_json, str):
            st.download_button(
                "Descargar JSON",
                data=st.session_state.export_json,
                file_name="transcript.json",
                mime="application/json",
                use_container_width=True,
            )

        st.divider()

        st.markdown("#### Models")

        llm_name = (
            os.getenv("LLM_SERVED_NAME")
            or os.getenv("LLM_MODEL_ID")
            or "custom-llm-model"
        )
        emb_name = (
            os.getenv("EMB_SERVED_NAME")
            or os.getenv("EMB_MODEL_ID")
            or "custom-embedding-model"
        )

        st.text_input("Model Name", value=llm_name, disabled=True, key="planner_model_name_display")
        st.caption("Embedding Server")
        st.text_input("Emb Name", value=emb_name, disabled=True, key="emb_model_name_display")
