import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


def _discover_db_candidates() -> List[str]:
    candidates: List[str] = []
    cwd = Path(os.getcwd())

    env_db = os.getenv("AGNOSTIC_DB_PATH") or os.getenv("VECTOR_DB_PATH")
    if env_db:
        candidates.append(str(Path(env_db)))

    candidates.append(str(cwd / "session" / "embeddings.db"))
    candidates.append(str(cwd / "embeddings.db"))
    candidates.append("/content/session/embeddings.db")

    # Descubrir DBs de embeddings en el workspace.
    for p in cwd.rglob("embeddings.db"):
        candidates.append(str(p))

    # Deduplicar manteniendo orden.
    seen = set()
    out: List[str] = []
    for c in candidates:
        n = str(Path(c))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _active_db_path() -> str:
    candidates = _discover_db_candidates()
    selected = st.session_state.get("active_db_path")
    if selected:
        return str(Path(selected))
    return candidates[0] if candidates else str(Path(os.getcwd()) / "session" / "embeddings.db")


def render_offline_tab(agent_factory):
    agent = agent_factory()

    tabs_labels = ["Knowledge Manager", "Tools Manager", "Skills Manager", "Logs de Ejecucion"]

    ui_plugins = []
    if "plugin_manager" in st.session_state and st.session_state.plugin_manager is not None:
        ui_plugins = st.session_state.plugin_manager.get_ui_plugins()
        ui_plugins = [p for p in ui_plugins if "offline" in p.type or p.type == "ui.panel"]

    for p in ui_plugins:
        tabs_labels.append(f"Plugin: {p.name}")

    tabs = st.tabs(tabs_labels)

    tab_km = tabs[0]
    tab_tm = tabs[1]
    tab_skills = tabs[2]
    tab_logs = tabs[3]
    plugin_tabs = tabs[4:]

    with tab_km:
        st.markdown("### Gestor de Conocimiento")

        if st.button("Reiniciar Conexion (Recargar Agente)"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.info("Sube PDFs, selecciona la DB correcta y revisa metadata por elemento.")

        candidates = _discover_db_candidates()
        default_path = _active_db_path()
        if default_path not in candidates:
            candidates = [default_path] + candidates

        selected_db = st.selectbox(
            "Base vectorial (.db)",
            options=candidates,
            index=max(0, candidates.index(default_path)),
            help="Selecciona el embeddings.db que quieres consultar e ingerir.",
        )
        custom_db = st.text_input("O ruta manual a DB", value="", placeholder="C:/ruta/a/embeddings.db")
        db_path = str(Path(custom_db.strip())) if custom_db.strip() else str(Path(selected_db))
        st.session_state["active_db_path"] = db_path

        # Unificar ruta de DB para tools y UI.
        os.environ["AGNOSTIC_DB_PATH"] = db_path
        os.environ["VECTOR_DB_PATH"] = db_path

        docs_dir = os.getenv("AGNOSTIC_DOCS_DIR", os.path.join(os.getcwd(), "documents"))
        os.makedirs(docs_dir, exist_ok=True)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        exists = os.path.exists(db_path)
        st.caption(f"DB activa: `{db_path}`")
        if exists:
            st.success("DB encontrada.")
        else:
            st.warning("La DB aun no existe. Se creara al ingerir el primer PDF.")

        from agnostic_agent.knowledge.vector import get_chunks_metadata, get_ingested_files, get_stats, ingest_pdf_file

        if exists:
            stats = get_stats(db_path)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Chunks", int(stats.get("chunks", 0)))
            col2.metric("Files", int(stats.get("files", 0)))
            col3.metric("Vectors", int(stats.get("vector_count", 0)))
            col4.metric("DB size (bytes)", int(stats.get("size_bytes", 0)))

            st.markdown("#### Archivos ingeridos")
            files_rows = get_ingested_files(db_path)
            if files_rows:
                st.dataframe(files_rows, use_container_width=True, hide_index=True)
            else:
                st.info("Sin registros en files_meta.")

            st.markdown("#### Metadata por elemento (chunks_meta)")
            max_rows = st.slider("Filas a mostrar", min_value=20, max_value=500, value=120, step=20)
            chunk_rows = get_chunks_metadata(db_path, limit=max_rows)
            if chunk_rows:
                st.dataframe(chunk_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No hay elementos en chunks_meta.")

        uploaded_file = st.file_uploader("Subir documento PDF", type=["pdf"])
        file_description = st.text_input("Descripcion", placeholder="Ej: Manual 2024")

        if uploaded_file:
            save_path = os.path.abspath(os.path.join(docs_dir, uploaded_file.name))
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            st.success(f"Archivo listo: `{uploaded_file.name}`")

            if st.button("Procesar e Ingestar", type="primary"):
                progress_bar = st.progress(0, text="Iniciando...")

                def _update_ui(p, msg):
                    progress_bar.progress(int(p * 100), text=msg)

                try:
                    result = ingest_pdf_file(
                        pdf_path=save_path,
                        db_path=db_path,
                        description=file_description,
                        progress_callback=_update_ui,
                    )
                    if result.get("success"):
                        st.success(f"Ingestion completada: {result['chunks']} chunks creados.")
                        st.json(result)
                    else:
                        st.error(f"Error: {result.get('error')}")
                except Exception as e:
                    st.error(f"Error critico durante la ingestion: {e}")

    with tab_tm:
        st.markdown("### Tools Playground")
        st.info("Prueba las herramientas disponibles con entradas manuales.")

        _tools_list = []
        if agent and hasattr(agent, "tools"):
            _tools_list = agent.tools

        tools_map = {t.name: t for t in _tools_list} if _tools_list else {}
        if not tools_map:
            st.warning("No tools loaded in agent.")
        else:
            groups: Dict[str, List[Any]] = {}
            for tname, tool in tools_map.items():
                prefix = tname.split(".")[0] if "." in tname else "General"
                groups.setdefault(prefix, []).append(tool)

            selected_group = st.selectbox("Grupo", list(groups.keys()))
            tools_in_group = groups.get(selected_group, [])
            selected_tool_name = st.selectbox("Herramienta", [t.name for t in tools_in_group])

            if selected_tool_name:
                tool = tools_map[selected_tool_name]
                st.markdown(f"### {tool.name}")

                doc_content = getattr(tool.func, "__doc__", None) if hasattr(tool, "func") else None
                if doc_content:
                    st.markdown(doc_content)
                elif tool.description:
                    st.markdown(tool.description)

                st.markdown("#### Inputs")
                args_schema = tool.args
                inputs: Dict[str, Any] = {}

                if args_schema:
                    for field_name, field_def in args_schema.items():
                        ftype = field_def.get("type", "string")
                        title = field_def.get("title", field_name)

                        if ftype == "integer":
                            inputs[field_name] = st.number_input(f"{title} ({field_name})", value=0, step=1)
                        elif ftype == "number":
                            inputs[field_name] = st.number_input(f"{title} ({field_name})", value=0.0)
                        elif ftype == "boolean":
                            inputs[field_name] = st.checkbox(f"{title} ({field_name})")
                        elif ftype == "array":
                            val_str = st.text_area(f"{title} ({field_name}) - JSON List", value="[]")
                            try:
                                inputs[field_name] = json.loads(val_str)
                            except Exception:
                                st.error(f"Invalid JSON for {field_name}")
                        else:
                            inputs[field_name] = st.text_input(f"{title} ({field_name})")

                if st.button(f"Ejecutar {selected_tool_name}", type="primary"):
                    try:
                        with st.spinner("Ejecutando..."):
                            output = tool.invoke(inputs)
                        st.markdown("#### Resultado")
                        st.success("Ejecucion exitosa")
                        st.write(output)

                        if "tool_logs" not in st.session_state:
                            st.session_state.tool_logs = []

                        st.session_state.tool_logs.append(
                            {
                                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                                "tool": selected_tool_name,
                                "inputs": inputs,
                                "output": str(output),
                            }
                        )
                    except Exception as e:
                        st.error(f"Error ejecutando tool: {e}")

    with tab_skills:
        st.markdown("### Gestor de Skills")
        if agent and agent.skill_registry:
            skills = agent.skill_registry.list_skills(enabled_only=False)
            if not skills:
                st.warning("No skills found.")
            else:
                col1, col2 = st.columns(2)
                for i, skill in enumerate(skills):
                    with col1 if i % 2 == 0 else col2:
                        st.markdown(f"**{skill.name}**")
                        if getattr(skill, "instructions", None):
                            with st.expander("Ver instrucciones completas", expanded=False):
                                st.markdown(skill.instructions)
                        elif skill.description:
                            st.caption(skill.description)

                        is_on = st.toggle("Habilitado", value=skill.enabled, key=f"s_{skill.name}")
                        if is_on != skill.enabled:
                            agent.skill_registry.set_enabled(skill.name, is_on)
                            if "skills_config" not in st.session_state:
                                st.session_state.skills_config = {}
                            st.session_state.skills_config[skill.name] = is_on
                            st.rerun()
                        st.divider()
        else:
            st.warning("Skill registry not available.")

    with tab_logs:
        st.markdown("### Logs")
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

    for i, plugin in enumerate(ui_plugins):
        with plugin_tabs[i]:
            st.markdown(f"### {plugin.name}")
            try:
                plugin.render(context=st)
            except Exception as e:
                st.error(f"Error rendering plugin {plugin.name}: {e}")
