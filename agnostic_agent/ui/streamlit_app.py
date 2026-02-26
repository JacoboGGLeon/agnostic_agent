from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from agnostic_agent.agent import Agent
from agnostic_agent.capabilities import PlannerConfig
from agnostic_agent.ui.panels.sidebar import render_sidebar
from agnostic_agent.ui.panels.online import render_online_tab
from agnostic_agent.ui.panels.offline import render_offline_tab
from agnostic_agent.plugins.manager import PluginManager
from agnostic_agent.config.loader import load_config

# -----------------------------
# Page Config
# -----------------------------
def _clean_app_title(value: str) -> str:
    raw = value if isinstance(value, str) else str(value or "")
    cleaned = re.sub(r"<[^>]+>", "", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "Agentic Lab"


APP_TITLE = _clean_app_title(os.getenv("AGNOSTIC_APP_TITLE", "Agentic Lab"))
APP_LOGO = os.getenv("AGNOSTIC_BRAND_LOGO_URL", "").strip()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_LOGO or "R",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# CSS Loading
# -----------------------------
def load_css() -> None:
    file_name = "styles.css"
    cwd = Path.cwd()
    this_dir = Path(__file__).resolve().parent
    candidates = [
        this_dir / "assets" / file_name,
        cwd / "assets" / file_name,
        cwd / "agnostic_agent" / "ui" / "assets" / file_name,
        cwd / "agnostic_agent" / "agnostic_agent" / "ui" / "assets" / file_name,
        cwd / "ui" / "assets" / file_name,
    ]

    for p in cwd.rglob(file_name):
        if len(p.relative_to(cwd).parts) <= 4:
            candidates.append(p)

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for c in candidates:
        cp = str(c.resolve()) if c.exists() else str(c)
        if cp not in seen:
            seen.add(cp)
            unique_candidates.append(c)

    for css_path in unique_candidates:
        if not css_path.exists():
            continue
        try:
            css_content = css_path.read_text(encoding="utf-8")
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            return
        except OSError:
            continue

    st.markdown(
        """
        <style>
        .topbar { padding: 10px; border-bottom: 1px solid #333; }
        .logo-img { width: 108px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    tried = ", ".join(str(p) for p in unique_candidates[:3])
    st.warning(f"CSS not found. Using minimal fallback. (Tried: {tried}...)")

load_css()

# -----------------------------
# Session State Init
# -----------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "tools_strict"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "msg_counter" not in st.session_state:
    st.session_state.msg_counter = 0
if "selected_msg_id" not in st.session_state:
    st.session_state.selected_msg_id = None
if "export_json" not in st.session_state:
    st.session_state.export_json = None
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

# Init Plugin Manager
if "plugin_manager" not in st.session_state:
    try:
        # Load config
        config = load_config()
        pm = PluginManager(config.plugins.model_dump())
        pm.load_plugins()
        st.session_state.plugin_manager = pm
    except Exception as e:
        st.error(f"Failed to initialize PluginManager: {e}")
        st.session_state.plugin_manager = None

# -----------------------------
# Helper: Agent Factory
# -----------------------------
def get_or_init_agent() -> Agent:
    if st.session_state.agent is None:
        with st.spinner(f"Inicializando agente..."):
            try:
                # Use environment variables or defaults
                # Note: We don't read from st.session_state inputs here to keep it simple for now,
                # relying on env vars or default config.
                
                cfg = PlannerConfig(
                    model_name=os.getenv("LLM_SERVED_NAME"),
                    temperature=0.0,
                    max_steps=16,
                )
                
                # Initializes Agent with default config or override
                st.session_state.agent = Agent.init(config_or_setup=cfg)

                # Sync Skills Config if previously set
                if "skills_config" in st.session_state and st.session_state.agent and st.session_state.agent.skill_registry:
                    for sname, senabled in st.session_state.skills_config.items():
                        st.session_state.agent.skill_registry.set_enabled(sname, senabled)

            except Exception as e:
                st.error(f"Error iniciando agente: {e}")
                st.stop()
    return st.session_state.agent

# -----------------------------
# Layout Construction
# -----------------------------

# 1. Sidebar
render_sidebar()

# 1.1 Dynamic theme override
if st.session_state.get("theme_mode", "dark") == "light":
    st.markdown(
        """
<style>
.stApp {
  background: #f7f9fc !important;
  color: #0d1b2a !important;
}
.topbar-offset {
  height: 0.35rem !important;
}
.stApp p, .stApp span, .stApp label, .stApp li, .stApp small, .stApp strong, .stApp em, .stApp pre {
  color: #0d1b2a !important;
}
section[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid #d7dce3 !important;
}
.stSidebar, .stSidebar * {
  color: #0d1b2a !important;
}
.topbar {
  background: #ffffff !important;
  border-color: #d7dce3 !important;
  box-shadow: 0 6px 20px rgba(0,0,0,.08) !important;
}
.title, .subtitle, .badge {
  color: #0d1b2a !important;
}
.stTabs [role="tab"] {
  color: #0d1b2a !important;
}
.stMarkdown, .stCaption, .stText, .stAlert, .stSelectbox label, .stRadio label, .stCheckbox label, .stTextInput label, .stTextArea label {
  color: #0d1b2a !important;
}
.stApp code, .stApp :not(pre) > code {
  color: #0b1f3a !important;
  background: #e9eff8 !important;
  border: 1px solid #d2dceb !important;
  border-radius: 6px !important;
  padding: 0.1rem 0.3rem !important;
}
[data-testid="stToolbar"] *, header[data-testid="stHeader"] * {
  color: #dbe7ff !important;
}
.stApp .stButton > button,
.stApp .stDownloadButton > button,
.stApp button[kind="secondary"] {
  background: #0a4fb3 !important;
  color: #ffffff !important;
  border: 1px solid #0a4fb3 !important;
}
.stApp .stButton > button:hover,
.stApp .stDownloadButton > button:hover,
.stApp button[kind="secondary"]:hover {
  background: #083e8d !important;
  border-color: #083e8d !important;
  color: #ffffff !important;
}
.stApp .stButton > button:focus,
.stApp .stDownloadButton > button:focus,
.stApp button[kind="secondary"]:focus {
  box-shadow: 0 0 0 0.2rem rgba(10, 79, 179, 0.28) !important;
}
[data-testid="stBaseButton-secondary"] * {
  color: #ffffff !important;
}
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] * {
  color: #0d1b2a !important;
  background-color: #ffffff !important;
}
.stChatInput [data-baseweb="textarea"] textarea {
  color: #0d1b2a !important;
  background-color: #ffffff !important;
}
.bubble-user {
  border-color: rgba(96, 61, 186, .45) !important;
  background: linear-gradient(180deg, rgba(96, 61, 186, .12), rgba(255, 255, 255, .8)) !important;
  color: #2f1c58 !important;
}
.bubble-agent {
  border-color: rgba(237, 139, 0, .6) !important;
  background: linear-gradient(180deg, rgba(237, 139, 0, .14), rgba(255, 255, 255, .9)) !important;
  color: #4a2c00 !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: #ffffff !important;
  border-color: #d7dce3 !important;
  box-shadow: 0 8px 26px rgba(0,0,0,.08) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  display: none !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

# 2. Topbar
st.markdown('<div class="topbar-offset"></div>', unsafe_allow_html=True)
logo_html = (
    f'<img class="logo-img" src="{APP_LOGO}" alt="brand" style="width: 108px; height: auto;"/>'
    if APP_LOGO
    else ""
)
st.markdown(
    f"""
<div class="topbar">
  <div class="brand">
    {logo_html}
    <div class="title">{APP_TITLE} · Studio</div>
  </div>
  <div class="badges">
    <!-- Badges here if needed -->
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# 3. Main Tabs
tab_online, tab_offline = st.tabs(["Online Chat", "Offline Manager"])

with tab_online:
    render_online_tab(get_or_init_agent)

with tab_offline:
    render_offline_tab(get_or_init_agent)

# force reload
