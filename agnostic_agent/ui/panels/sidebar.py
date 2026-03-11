import html
import json
import os
import re

import streamlit as st

DEFAULT_BRAND_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/BBVA_2019.svg/1280px-BBVA_2019.svg.png"


def _resolve_llm_name() -> str:
    for env_key in ("OPENAI_MODEL", "AGNOSTIC_LLM_MODEL", "LLM_MODEL_ID", "LLM_SERVED_NAME"):
        val = os.getenv(env_key, "")
        if val.strip():
            return val.strip()

    agent = st.session_state.get("agent")
    if agent is not None:
        cfg = getattr(agent, "planner_config", None)
        model_name = getattr(cfg, "model_name", None) if cfg is not None else None
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()

    return "custom-llm-model"


def _resolve_emb_name() -> str:
    for env_key in ("OPENAI_EMBED_MODEL", "AGNOSTIC_EMB_MODEL", "EMB_MODEL_ID", "EMB_SERVED_NAME"):
        val = os.getenv(env_key, "")
        if val.strip():
            return val.strip()

    agent = st.session_state.get("agent")
    if agent is not None:
        setup_cfg = getattr(agent, "setup_config", {}) or {}
        models_cfg = setup_cfg.get("models", {}) if isinstance(setup_cfg, dict) else {}
        emb_cfg = models_cfg.get("emb", {}) if isinstance(models_cfg, dict) else {}
        for key in ("served_name", "model", "name"):
            val = emb_cfg.get(key) if isinstance(emb_cfg, dict) else None
            if isinstance(val, str) and val.strip():
                return val.strip()

    return "custom-embedding-model"


def _clean_ui_text(value: str, default: str = "") -> str:
    raw = value if isinstance(value, str) else str(value or "")
    raw = html.unescape(raw).replace("`", " ")
    cleaned = re.sub(r"<[^>]+>", " ", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*[·\-|]\s*(studio|settings)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or default


def render_sidebar() -> None:
    app_title = _clean_ui_text(os.getenv("AGNOSTIC_APP_TITLE", "Agentic Lab"), "Agentic Lab")
    logo_url = os.getenv("AGNOSTIC_BRAND_LOGO_URL", "").strip() or DEFAULT_BRAND_LOGO_URL

    with st.sidebar:
        st.image(logo_url, width=140)
        st.markdown(f"### {app_title} · Settings")

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

        st.toggle(
            "Historial en conversacion",
            value=True,
            key="conversation_history_enabled",
            help="Si se desactiva, cada turno se ejecuta sin arrastrar mensajes previos del chat.",
        )

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

        llm_name = _clean_ui_text(_resolve_llm_name(), "custom-llm-model")
        emb_name = _clean_ui_text(_resolve_emb_name(), "custom-embedding-model")

        st.text_input("Model Name", value=llm_name, disabled=True, key="planner_model_name_display")
        st.caption("Embedding Server")
        st.text_input("Emb Name", value=emb_name, disabled=True, key="emb_model_name_display")
