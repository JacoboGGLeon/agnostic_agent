import streamlit as st
import os
import json
from typing import Dict, Any, List
from agnostic_agent.plugins.manager import PluginManager

def render_offline_tab(agent_factory):
    agent = agent_factory()
    
    # Discovery of UI plugins
    # We need a PluginManager instance. Since we don't have a global one easily accessible 
    # (except maybe if we attach it to st.session_state or Agent), we'll instantiate one 
    # based on the current config. Ideally, this should be a singleton or passed down.
    # For now, let's try to get it from session state or init it.
    
    if "plugin_manager" not in st.session_state:
        # Load config to init manager
        # Assuming config is loaded in environment or we can load it here
        # For this refactor, we'll try to load a minimal manager or check if Agent has it (Agent facade doesn't expose it yet)
        # TODO: Refactor Agent to expose plugin_manager
        
        # Temporary: Load dummy manager or from file if possible. 
        # Better: Add plugin_manager to st.session_state during app init.
        pass

    # Standard Tabs
    tabs_labels = ["📚 Knowledge Manager", "🛠 Tools Manager", "🧩 Skills Manager", "📜 Logs de Ejecución"]
    
    # Add Plugin Tabs
    # We need a way to get plugins. Let's assume we can get them from a global registry or helper.
    # For now, we will just use a placeholder list or try to use `st.session_state.plugin_manager` if exists.
    
    ui_plugins = []
    if "plugin_manager" in st.session_state:
        ui_plugins = st.session_state.plugin_manager.get_ui_plugins() # All UI plugins
        # Filter for offline tabs?
        # Contract: type="ui.tab.offline" or just "ui.tab"
        ui_plugins = [p for p in ui_plugins if "offline" in p.type or p.type == "ui.panel"]

    for p in ui_plugins:
        tabs_labels.append(f"🔌 {p.name}")

    tabs = st.tabs(tabs_labels)
    
    # Unpack tabs
    tab_km = tabs[0]
    tab_tm = tabs[1]
    tab_skills = tabs[2]
    tab_logs = tabs[3]
    
    plugin_tabs = tabs[4:]

    # 📚 Knowledge Manager
    with tab_km:
        st.markdown("### 📚 Gestor de Conocimiento")
        st.info("Subir documentos PDF para procesarlos.")
        
        uploaded_file = st.file_uploader("Subir documento PDF", type=["pdf"])
        file_description = st.text_input("Descripción", placeholder="Ej: Manual 2024")
        
        DOCS_DIR = os.getenv("AGNOSTIC_DOCS_DIR", os.path.join(os.getcwd(), "documents"))
        DB_PATH = os.getenv("AGNOSTIC_DB_PATH", os.path.join(os.getcwd(), "embeddings.db"))
        os.makedirs(DOCS_DIR, exist_ok=True)
        
        if uploaded_file:
            save_path = os.path.abspath(os.path.join(DOCS_DIR, uploaded_file.name))
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Guardado en: `{save_path}`")
            
            if st.button("🚀 Procesar", type="primary"):
                 st.info("Ingestion functionality is migrated. Please use CLI tools for now.")

    # 🛠 Tools Manager
    with tab_tm:
        st.markdown("### 🛠 Gestor de Herramientas")
        _tools_list = []
        if agent and hasattr(agent, "tools"):
             _tools_list = agent.tools
        
        tools_map = {t.name: t for t in _tools_list} if _tools_list else {}
        
        if not tools_map:
             st.warning("No tools loaded in agent.")
        else:
             for tname in tools_map:
                 st.text(f"- {tname}")

    # 🧩 Skills Manager
    with tab_skills:
        st.markdown("### 🧩 Gestor de Skills")
        if agent and agent.skill_registry:
            skills = agent.skill_registry.list_skills(enabled_only=False)
            if not skills:
                 st.warning("No skills found.")
            else:
                for skill in skills:
                    is_on = st.toggle(skill.name, value=skill.enabled, key=f"s_{skill.name}")
                    if is_on != skill.enabled:
                        agent.skill_registry.set_enabled(skill.name, is_on)
                        if "skills_config" not in st.session_state: st.session_state.skills_config = {}
                        st.session_state.skills_config[skill.name] = is_on
                        st.rerun()
        else:
            st.warning("Skill registry not available.")

    # 📜 Logs
    with tab_logs:
        st.markdown("### 📜 Logs")
        if st.button("Limpiar"):
            st.session_state["tool_logs"] = []
            st.rerun()
            
        logs = st.session_state.get("tool_logs", [])
        if not logs:
            st.info("No logs.")
        else:
            for l in reversed(logs):
                 with st.expander(f"{l.get('timestamp')} | {l.get('tool')}"):
                     st.json(l)
                     
    # 🔌 Plugins
    for i, plugin in enumerate(ui_plugins):
        with plugin_tabs[i]:
            st.markdown(f"### {plugin.name}")
            try:
                plugin.render(context=st)
            except Exception as e:
                st.error(f"Error rendering plugin {plugin.name}: {e}")
