import datetime
import json
import os
import sqlite3
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


def _discover_session_roots() -> List[Path]:
    candidates = [
        Path(os.getcwd()) / "session",
        Path("/content/session"),
        Path(os.getcwd()),
    ]
    out: List[Path] = []
    seen = set()
    for path in candidates:
        try:
            norm = str(path.resolve())
        except Exception:
            norm = str(path)
        if norm not in seen and path.exists():
            seen.add(norm)
            out.append(path)
    return out


def _discover_session_sources() -> List[Dict[str, Any]]:
    roots = _discover_session_roots()
    finance_names = {"contabilidad.db", "transacciones.db", "rules.md", "dictionary.md"}
    sources: List[Dict[str, Any]] = []
    seen_paths = set()

    for root in roots:
        for path in list(root.glob("*.db")) + list(root.glob("*.md")):
            norm = str(path.resolve())
            if norm in seen_paths:
                continue
            seen_paths.add(norm)
            reachable, detail = _check_source_reachability(path)
            sources.append(
                {
                    "name": path.name,
                    "kind": "db" if path.suffix.lower() == ".db" else "md",
                    "path": norm,
                    "size_kb": round(path.stat().st_size / 1024.0, 2),
                    "root": str(root.resolve()),
                    "reachable": reachable,
                    "semaforo": "GREEN" if reachable else "RED",
                    "reachability_detail": detail,
                    "finance_target": path.name in finance_names,
                }
            )

    return sorted(sources, key=lambda row: (not row["reachable"], row["kind"], row["name"]))


def _check_source_reachability(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "path does not exist"

    if path.suffix.lower() == ".db":
        try:
            conn = sqlite3.connect(str(path))
            cur = conn.cursor()
            cur.execute("SELECT 1")
            conn.close()
            return True, "sqlite open ok"
        except Exception as exc:
            return False, f"sqlite open error: {exc}"

    if path.suffix.lower() == ".md":
        try:
            _ = path.read_text(encoding="utf-8", errors="ignore")
            return True, "markdown read ok"
        except Exception as exc:
            return False, f"markdown read error: {exc}"

    return False, "unsupported file type"


def _db_inspect(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    for (table_name,) in tables:
        try:
            count = cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        except Exception:
            count = None
        rows.append({"table": table_name, "rows": count})
    conn.close()
    return rows


def _read_md_preview(path: str, max_chars: int = 2500) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n... (preview truncado)"
    return text


def _sqlite_quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sqlite_list_tables(path: str) -> List[str]:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _sqlite_table_schema(path: str, table_name: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    q = f"PRAGMA table_info({_sqlite_quote_ident(table_name)})"
    rows = cur.execute(q).fetchall()
    conn.close()
    out: List[Dict[str, Any]] = []
    for cid, name, col_type, notnull, default_value, pk in rows:
        out.append(
            {
                "cid": cid,
                "name": name,
                "type": col_type,
                "not_null": bool(notnull),
                "default": default_value,
                "pk": bool(pk),
            }
        )
    return out


def _sqlite_table_rows(path: str, table_name: str, limit: int, offset: int = 0) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    q = (
        f"SELECT * FROM {_sqlite_quote_ident(table_name)} "
        f"LIMIT {int(limit)} OFFSET {int(offset)}"
    )
    rows = cur.execute(q).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _filter_sources_for_skill(skill_name: str, sources: List[Dict[str, Any]], active_vector_db: str) -> List[Dict[str, Any]]:
    if skill_name == "contabilidad_instantanea":
        out = [s for s in sources if s.get("finance_target")]
        return out

    # semantic_researcher: prefer embeddings DB + markdown docs if available.
    out: List[Dict[str, Any]] = []
    active_name = Path(active_vector_db).name.lower() if active_vector_db else ""
    for s in sources:
        name = str(s.get("name", "")).lower()
        if s.get("kind") == "db" and (name == "embeddings.db" or name == active_name):
            out.append(s)
        elif s.get("kind") == "md" and not s.get("finance_target"):
            out.append(s)

    # If none found for semantic, include active vector DB as fallback pseudo-source.
    if not any(x.get("kind") == "db" for x in out) and active_vector_db:
        p = Path(active_vector_db)
        if p.exists():
            reachable, detail = _check_source_reachability(p)
            out.insert(
                0,
                {
                    "name": p.name,
                    "kind": "db",
                    "path": str(p.resolve()),
                    "size_kb": round(p.stat().st_size / 1024.0, 2),
                    "root": str(p.parent.resolve()),
                    "reachable": reachable,
                    "semaforo": "GREEN" if reachable else "RED",
                    "reachability_detail": detail,
                    "finance_target": False,
                },
            )
    return out


def _render_sqlite_viewer(db_path: str, key_prefix: str) -> None:
    st.markdown("##### SQLite Viewer")
    try:
        tables = _sqlite_list_tables(db_path)
    except Exception as exc:
        st.error(f"No se pudo abrir la DB: {exc}")
        return

    if not tables:
        st.info("No hay tablas visibles en esta DB.")
        return

    selected_table = st.selectbox(
        "Tabla",
        options=tables,
        key=f"{key_prefix}_sqlite_table",
    )

    c1, c2 = st.columns(2)
    limit = c1.number_input("Filas", min_value=10, max_value=1000, value=100, step=10, key=f"{key_prefix}_limit")
    offset = c2.number_input("Offset", min_value=0, max_value=500000, value=0, step=50, key=f"{key_prefix}_offset")

    try:
        schema = _sqlite_table_schema(db_path, selected_table)
        st.caption("Esquema")
        st.dataframe(schema, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"No se pudo leer esquema de {selected_table}: {exc}")

    try:
        rows = _sqlite_table_rows(db_path, selected_table, int(limit), int(offset))
        st.caption("Datos")
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("La consulta no devolvio filas.")
    except Exception as exc:
        st.error(f"No se pudo leer filas de {selected_table}: {exc}")


def _render_markdown_pretty(md_path: str, key_prefix: str) -> None:
    st.markdown("##### Markdown Viewer")
    max_chars = st.slider(
        "Max chars preview",
        min_value=1000,
        max_value=30000,
        value=8000,
        step=500,
        key=f"{key_prefix}_md_chars",
    )
    try:
        md_text = _read_md_preview(md_path, max_chars=max_chars)
    except Exception as exc:
        st.error(f"No se pudo leer markdown: {exc}")
        return

    render_pretty = st.toggle("Render markdown pretty", value=True, key=f"{key_prefix}_md_pretty")
    if render_pretty:
        st.markdown(md_text)
    with st.expander("Ver markdown raw"):
        st.code(md_text, language="markdown")


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
            col4.metric("L2 Docs", int(stats.get("doc_index_count", 0)))

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

        st.markdown("#### Session Sources (DB + MD)")
        session_sources = _discover_session_sources()
        if session_sources:
            total_sources = len(session_sources)
            reachable_sources = sum(1 for row in session_sources if row.get("reachable"))
            c_ok, c_ko = st.columns(2)
            c_ok.metric("Reachable", reachable_sources)
            c_ko.metric("Unreachable", total_sources - reachable_sources)

            finance_map = {
                row["name"]: row["path"]
                for row in session_sources
                if row.get("finance_target") and row.get("reachable")
            }
            if st.button("Usar fuentes detectadas para Finance"):
                if "contabilidad.db" in finance_map:
                    os.environ["AGNOSTIC_FIN_ACC_DB"] = finance_map["contabilidad.db"]
                if "transacciones.db" in finance_map:
                    os.environ["AGNOSTIC_FIN_TRANS_DB"] = finance_map["transacciones.db"]
                if "rules.md" in finance_map:
                    os.environ["AGNOSTIC_FIN_RULES_MD"] = finance_map["rules.md"]
                if "dictionary.md" in finance_map:
                    os.environ["AGNOSTIC_FIN_DICT_MD"] = finance_map["dictionary.md"]
                st.success("Variables AGNOSTIC_FIN_* actualizadas con fuentes detectadas y reachables.")

            km_tabs = st.tabs(["General", "semantic_researcher", "contabilidad_instantanea"])

            with km_tabs[0]:
                st.dataframe(session_sources, use_container_width=True, hide_index=True)

                selected_source = st.selectbox(
                    "Inspeccionar fuente",
                    options=[row["path"] for row in session_sources],
                    format_func=lambda path: f"{Path(path).name} - {path}",
                    key="km_general_source",
                )
                selected_row = next((row for row in session_sources if row["path"] == selected_source), None)
                if selected_row and selected_row["kind"] == "db":
                    _render_sqlite_viewer(selected_source, key_prefix="km_general")
                elif selected_row:
                    _render_markdown_pretty(selected_source, key_prefix="km_general")

            with km_tabs[1]:
                st.markdown("##### Fuentes por skill: `semantic_researcher`")
                semantic_sources = _filter_sources_for_skill("semantic_researcher", session_sources, db_path)
                if semantic_sources:
                    st.dataframe(semantic_sources, use_container_width=True, hide_index=True)
                    sem_db_sources = [s for s in semantic_sources if s.get("kind") == "db"]
                    sem_md_sources = [s for s in semantic_sources if s.get("kind") == "md"]

                    if sem_db_sources:
                        sem_db_path = st.selectbox(
                            "DB para semantic_researcher",
                            options=[s["path"] for s in sem_db_sources],
                            format_func=lambda path: f"{Path(path).name} - {path}",
                            key="km_sem_db",
                        )
                        _render_sqlite_viewer(sem_db_path, key_prefix="km_sem")
                    else:
                        st.info("No se detecto DB para semantic_researcher.")

                    if sem_md_sources:
                        sem_md_path = st.selectbox(
                            "Markdown para semantic_researcher",
                            options=[s["path"] for s in sem_md_sources],
                            format_func=lambda path: f"{Path(path).name} - {path}",
                            key="km_sem_md",
                        )
                        _render_markdown_pretty(sem_md_path, key_prefix="km_sem")
                else:
                    st.info("No se detectaron fuentes para semantic_researcher.")

            with km_tabs[2]:
                st.markdown("##### Fuentes por skill: `contabilidad_instantanea`")
                fin_sources = _filter_sources_for_skill("contabilidad_instantanea", session_sources, db_path)
                if fin_sources:
                    st.dataframe(fin_sources, use_container_width=True, hide_index=True)
                    fin_db_sources = [s for s in fin_sources if s.get("kind") == "db"]
                    fin_md_sources = [s for s in fin_sources if s.get("kind") == "md"]

                    if fin_db_sources:
                        fin_db_path = st.selectbox(
                            "DB para contabilidad_instantanea",
                            options=[s["path"] for s in fin_db_sources],
                            format_func=lambda path: f"{Path(path).name} - {path}",
                            key="km_fin_db",
                        )
                        _render_sqlite_viewer(fin_db_path, key_prefix="km_fin")
                    else:
                        st.info("No se detecto DB para contabilidad_instantanea.")

                    if fin_md_sources:
                        fin_md_path = st.selectbox(
                            "Markdown para contabilidad_instantanea",
                            options=[s["path"] for s in fin_md_sources],
                            format_func=lambda path: f"{Path(path).name} - {path}",
                            key="km_fin_md",
                        )
                        _render_markdown_pretty(fin_md_path, key_prefix="km_fin")
                else:
                    st.info("No se detectaron fuentes para contabilidad_instantanea.")
        else:
            st.info("No se detectaron fuentes de session (.db/.md) en rutas comunes.")

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
            def _tool_group_name(tool_obj: Any) -> str:
                fn = getattr(tool_obj, "func", None)
                module_name = getattr(fn, "__module__", "") if fn else ""
                if ".tools." in module_name:
                    return module_name.split(".tools.")[-1]
                return "misc"

            def _example_payload(tool_name: str) -> Dict[str, Any]:
                examples: Dict[str, Dict[str, Any]] = {
                    "is_palindrome": {"text": "Anita lava la tina"},
                    "word_count": {"text": "OpenAI Gym es un toolkit de RL"},
                    "to_upper": {"text": "hola mundo"},
                    "eval_math_expression": {"expression": "(10 - 4) / 2"},
                    "sum_numbers": {"numbers": [1, 2, 3]},
                    "average_numbers": {"numbers": [4, 6, 8]},
                    "search_knowledge_base": {"query": "open ai gym", "top_k": 10},
                    "list_knowledge_sources": {},
                    "embed_texts": {"texts": ["open ai gym", "reinforcement learning"]},
                    "semantic_search_in_memory": {
                        "query": "open ai gym",
                        "documents": [
                            "OpenAI Gym is a toolkit for reinforcement learning.",
                            "Breiman discusses statistical modeling cultures.",
                        ],
                        "top_k": 2,
                    },
                    "context_search_in_csv": {
                        "query": "open ai gym",
                        "csv_path": "2026-02-19T22-21_export.csv",
                        "text_columns": ["file", "description"],
                        "top_k": 5,
                    },
                    "embed_context_tables": {
                        "table_paths": ["2026-02-19T22-21_export.csv"],
                        "text_columns": {
                            "2026-02-19T22-21_export.csv": ["file", "description"]
                        },
                    },
                    "rerank_docs": {
                        "query": "open ai gym",
                        "documents": [
                            {"content": "OpenAI Gym toolkit for RL experiments"},
                            {"content": "The Two Cultures by Leo Breiman"},
                        ],
                    },
                    "judge_row_with_context": {
                        "row": {"id": "row_1", "term": "OpenAI Gym"},
                        "param_hits": [{"text": "OpenAI Gym toolkit"}],
                        "glossary_hits": [],
                    },
                }
                return examples.get(tool_name, {})

            def _normalized_field_type(
                field_name: str,
                field_def: Dict[str, Any],
                example_value: Any,
            ) -> str:
                if isinstance(example_value, bool):
                    return "boolean"
                if isinstance(example_value, int) and not isinstance(example_value, bool):
                    return "integer"
                if isinstance(example_value, float):
                    return "number"
                if isinstance(example_value, list):
                    return "array"
                if isinstance(example_value, dict):
                    return "object"

                ftype = (field_def or {}).get("type", "string")
                if field_name in ("text_columns", "row") and ftype == "string":
                    return "object"
                return ftype

            def _tool_markdown_doc(tool_obj: Any) -> str:
                name = tool_obj.name
                args_schema = tool_obj.args or {}
                example = _example_payload(name)
                input_lines = []
                for field_name, field_def in args_schema.items():
                    ex_value = example.get(field_name)
                    ftype = _normalized_field_type(field_name, field_def, ex_value)
                    input_lines.append(f"- `{field_name}`: `{ftype}`")
                if not input_lines:
                    input_lines.append("- _(sin parametros)_")

                output_hint = "Ver salida real de la tool."
                if name == "is_palindrome":
                    output_hint = "`true` si el texto es palindromo, si no `false`."

                ex = json.dumps(example, ensure_ascii=False, indent=2) if example else "{}"
                return (
                    "#### Test Tool\n"
                    f"**input:**\n{chr(10).join(input_lines)}\n\n"
                    f"**output:**\n- {output_hint}\n\n"
                    f"**ejemplo:**\n```json\n{ex}\n```"
                )

            groups: Dict[str, List[Any]] = {}
            for tname, tool in tools_map.items():
                gname = _tool_group_name(tool)
                groups.setdefault(gname, []).append(tool)

            ordered_groups = sorted(groups.keys())
            selected_group = st.selectbox("Grupo", ["General (Todas)"] + ordered_groups)
            if selected_group == "General (Todas)":
                tools_in_group = sorted(tools_map.values(), key=lambda t: t.name)
            else:
                tools_in_group = sorted(groups.get(selected_group, []), key=lambda t: t.name)
            selected_tool_name = st.selectbox("Herramienta", [t.name for t in tools_in_group])

            if selected_tool_name:
                tool = tools_map[selected_tool_name]
                st.markdown(f"### {tool.name}")

                doc_content = getattr(tool.func, "__doc__", None) if hasattr(tool, "func") else None
                if doc_content:
                    st.markdown(doc_content)
                elif tool.description:
                    st.markdown(tool.description)

                st.markdown(_tool_markdown_doc(tool))
                st.markdown("#### Test Tool")
                args_schema = tool.args
                inputs: Dict[str, Any] = {}
                example_payload = _example_payload(tool.name)

                if args_schema:
                    for field_name, field_def in args_schema.items():
                        default_value = example_payload.get(field_name)
                        ftype = _normalized_field_type(field_name, field_def, default_value)
                        title = field_def.get("title", field_name)

                        if ftype == "integer":
                            initial = int(default_value) if isinstance(default_value, (int, float)) else 0
                            inputs[field_name] = st.number_input(f"{title} ({field_name})", value=initial, step=1)
                        elif ftype == "number":
                            initial = float(default_value) if isinstance(default_value, (int, float)) else 0.0
                            inputs[field_name] = st.number_input(f"{title} ({field_name})", value=initial)
                        elif ftype == "boolean":
                            initial = bool(default_value) if isinstance(default_value, bool) else False
                            inputs[field_name] = st.checkbox(f"{title} ({field_name})", value=initial)
                        elif ftype == "array":
                            initial = default_value if isinstance(default_value, list) else []
                            val_str = st.text_area(
                                f"{title} ({field_name}) - JSON List",
                                value=json.dumps(initial, ensure_ascii=False),
                            )
                            try:
                                inputs[field_name] = json.loads(val_str)
                            except Exception:
                                st.error(f"Invalid JSON for {field_name}")
                        elif ftype == "object":
                            initial = default_value if isinstance(default_value, dict) else {}
                            val_str = st.text_area(
                                f"{title} ({field_name}) - JSON Object",
                                value=json.dumps(initial, ensure_ascii=False, indent=2),
                            )
                            try:
                                inputs[field_name] = json.loads(val_str)
                            except Exception:
                                st.error(f"Invalid JSON for {field_name}")
                        else:
                            initial = str(default_value) if default_value is not None else ""
                            inputs[field_name] = st.text_input(f"{title} ({field_name})", value=initial)

                if st.button("Test Tool", type="primary"):
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
                        
                        # Render YAML Metadata nicely
                        if skill.description:
                            st.caption(skill.description)
                        
                        if skill.tools:
                            st.markdown(f"🛠 **Tools**: {', '.join([f'`{t}`' for t in skill.tools])}")
                        
                        if skill.knowledge:
                            st.markdown(f"📚 **Knowledge**: {', '.join([f'`{k}`' for k in skill.knowledge])}")

                        if getattr(skill, "instructions", None):
                            with st.expander("Ver instrucciones completas", expanded=False):
                                st.markdown(skill.instructions)

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
