from __future__ import annotations

import os
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
st.set_page_config(
    page_title="Agentic Lab · BBVA",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/BBVA_2019.svg/1280px-BBVA_2019.svg.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# CSS Loading
# -----------------------------
def load_css():
    file_name = "styles.css"
    # Extended search strategy:
    # 1. Standard paths
    # 2. Recursive search in current directory (depth 3)
    
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", file_name),
        os.path.join(os.getcwd(), "assets", file_name),
        os.path.join(os.getcwd(), "agnostic_agent", "ui", "assets", file_name),
        "/content/assets/styles.css",
        "assets/styles.css",
    ]
    
    # Add recursive search
    try:
        for root, dirs, files in os.walk(os.getcwd()):
            if file_name in files:
                candidates.append(os.path.join(root, file_name))
            # Limit depth hack (not perfect but safe for small repos)
            if root.count(os.sep) - os.getcwd().count(os.sep) > 3:
                del dirs[:] 
    except:
        pass

    css_content = ""
    found_path = None
    
    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            unique_candidates.append(c)
            seen.add(c)
            
    for path in unique_candidates:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    css_content = f.read()
                found_path = path
                break
            except:
                continue
                
    if found_path:
        st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
    else:
        # Fallback inline minimal style
        st.markdown("""
        <style>
        .topbar { padding: 10px; border-bottom: 1px solid #333; }
        .logo-img { width: 108px; }
        </style>
        """, unsafe_allow_html=True)
        st.warning(f"⚠️ CSS not found. Using minimal fallback. (Tried: {', '.join(unique_candidates[:3])}...)")

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
body, p, span, label, li, small, strong, em, code, pre {
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
st.markdown(
    f"""
<div class="topbar">
  <div class="brand">
    <img class="logo-img" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/BBVA_2019.svg/1280px-BBVA_2019.svg.png" alt="BBVA" style="width: 108px; height: auto;"/>
    <div class="title">Agentic Lab · Studio</div>
  </div>
  <div class="badges">
    <!-- Badges here if needed -->
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# 3. Main Tabs
tab_online, tab_offline = st.tabs(["💬 Online Chat", "🛠 Offline Manager"])

with tab_online:
    render_online_tab(get_or_init_agent)

with tab_offline:
    render_offline_tab(get_or_init_agent)
