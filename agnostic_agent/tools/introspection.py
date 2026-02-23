from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import sys
import sqlite3
import re
import json

from langchain_core.tools import BaseTool

from agnostic_agent.skills import SkillRegistry
from agnostic_agent.tools.decorators import tool


def _default_skills_dir() -> str:
    env_dir = os.getenv("AGNOSTIC_SKILLS_DIR", "").strip()
    if env_dir:
        return env_dir
    # agnostic_agent/tools/introspection.py -> agnostic_agent/tools -> agnostic_agent/skills
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "skills")


def _iter_tools() -> List[BaseTool]:
    # Avoid importing agnostic_agent.tools (package __init__) to prevent circular imports.
    from agnostic_agent.tools import basic, math, semantic, finance

    modules = [basic, math, semantic, finance, sys.modules[__name__]]
    out: List[BaseTool] = []
    seen = set()

    for module in modules:
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, BaseTool):
                if obj.name in seen:
                    continue
                out.append(obj)
                seen.add(obj.name)
    return out


def _find_tool(name: str) -> Optional[BaseTool]:
    target = (name or "").strip()
    if not target:
        return None
    for t in _iter_tools():
        if t.name == target:
            return t
    return None


def _resolve_vector_db_path() -> str:
    # Keep consistent with agnostic_agent.tools.semantic._resolve_vector_db_path
    db_path = os.getenv("AGNOSTIC_DB_PATH") or os.getenv("VECTOR_DB_PATH")
    if db_path:
        return db_path

    default_session_db = os.path.join(os.getcwd(), "session", "embeddings.db")
    if os.path.exists(default_session_db):
        return default_session_db
    return os.path.join(os.getcwd(), "embeddings.db")


def _hkb_table_names(db_path: str) -> List[str]:
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


@tool(mode="public")
def list_skills(name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lista skills disponibles (name/description/tools/knowledge) desde `agnostic_agent/skills/*.md`.
    """
    reg = SkillRegistry(_default_skills_dir())
    flt = (name_filter or "").strip().lower()

    rows: List[Dict[str, Any]] = []
    for s in sorted(reg.list_skills(enabled_only=False), key=lambda x: x.name):
        if flt and flt not in s.name.lower() and flt not in (s.description or "").lower():
            continue
        rows.append(
            {
                "name": s.name,
                "description": s.description,
                "tools": list(s.tools or []),
                "knowledge": list(s.knowledge or []),
                "enabled": bool(getattr(s, "enabled", True)),
                "file": os.path.basename(s.file_path) if s.file_path else None,
            }
        )
    return rows


@tool(mode="public")
def read_skill(skill_name: str) -> Dict[str, Any]:
    """
    Lee una skill por nombre y devuelve metadata + instrucciones completas.
    """
    reg = SkillRegistry(_default_skills_dir())
    s = reg.get_skill(skill_name)
    if not s:
        return {"error": f"Skill not found: {skill_name}"}
    return {
        "name": s.name,
        "description": s.description,
        "tools": list(s.tools or []),
        "knowledge": list(s.knowledge or []),
        "enabled": bool(getattr(s, "enabled", True)),
        "file_path": s.file_path,
        "instructions": s.instructions,
    }


@tool(mode="public")
def list_tools(name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lista tools disponibles (name/description/args).
    """
    flt = (name_filter or "").strip().lower()
    rows: List[Dict[str, Any]] = []

    for t in sorted(_iter_tools(), key=lambda x: x.name):
        desc = getattr(t, "description", "") or ""
        if flt and flt not in t.name.lower() and flt not in desc.lower():
            continue
        args = getattr(t, "args", None)
        meta = getattr(t, "_agnostic_metadata", None)
        rows.append(
            {
                "name": t.name,
                "description": desc,
                "args": args if isinstance(args, dict) else None,
                "mode": (meta or {}).get("mode") if isinstance(meta, dict) else None,
            }
        )
    return rows


@tool(mode="public")
def read_tool(tool_name: str) -> Dict[str, Any]:
    """
    Devuelve detalle de una tool: description, args y metadata agnóstica.
    """
    t = _find_tool(tool_name)
    if not t:
        return {"error": f"Tool not found: {tool_name}"}

    meta = getattr(t, "_agnostic_metadata", None)
    args = getattr(t, "args", None)
    return {
        "name": t.name,
        "description": getattr(t, "description", "") or "",
        "args": args if isinstance(args, dict) else None,
        "mode": (meta or {}).get("mode") if isinstance(meta, dict) else None,
        "tags": list(getattr(t, "tags", []) or []),
    }


@tool(mode="public")
def hkb_status() -> Dict[str, Any]:
    """
    Inspecciona la Knowledge Base vectorial (sqlite-vec) y reporta si tiene el esquema HKB (docs_index + chunks).
    """
    db_path = _resolve_vector_db_path()
    tables = _hkb_table_names(db_path)
    required = {"v_chunks", "chunks_meta", "docs_index"}
    is_hkb = required.issubset(set(tables))

    stats: Dict[str, Any] = {}
    if os.path.exists(db_path):
        try:
            from agnostic_agent.knowledge.vector import get_stats

            stats = get_stats(db_path)
        except Exception as e:
            stats = {"warning": f"Failed to get_stats: {e}"}

    return {
        "db_path": db_path,
        "exists": os.path.exists(db_path),
        "tables": tables,
        "is_hkb_schema": is_hkb,
        "stats": stats,
    }


@tool(mode="public")
def list_hkb_documents() -> List[Dict[str, Any]]:
    """
    Lista documentos ingeridos en la HKB (si existe), usando files_meta.
    """
    db_path = _resolve_vector_db_path()
    if not os.path.exists(db_path):
        return []

    try:
        from agnostic_agent.knowledge.vector import get_ingested_files

        return get_ingested_files(db_path)
    except Exception as e:
        return [{"error": f"Failed to list HKB docs: {e}", "db_path": db_path}]


@tool(mode="public")
def list_knowledge(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lista knowledge entries (documentos indexados) dentro de una DB vectorial.
    """
    target_db = (db_path or "").strip() or _resolve_vector_db_path()
    if not os.path.exists(target_db):
        return [{"error": "DB not found", "db_path": target_db}]

    conn = sqlite3.connect(target_db)
    try:
        tables = set(_hkb_table_names(target_db))
        if "docs_index" not in tables:
            return [{"error": "docs_index table not found (not HKB schema)", "db_path": target_db}]

        desc_by_source: Dict[str, str] = {}
        if "files_meta" in tables:
            rows = conn.execute(
                "SELECT source_path, COALESCE(description, '') FROM files_meta"
            ).fetchall()
            for source_path, desc in rows:
                desc_by_source[str(source_path)] = str(desc or "")

        out: List[Dict[str, Any]] = []
        for source_path, n_chunks, updated_at in conn.execute(
            "SELECT source_path, n_chunks, updated_at FROM docs_index ORDER BY updated_at DESC"
        ).fetchall():
            source_key = str(source_path)
            out.append(
                {
                    "source_path": source_key,
                    "file": os.path.basename(source_key),
                    "description": desc_by_source.get(source_key, ""),
                    "n_chunks": int(n_chunks or 0),
                    "updated_at": updated_at,
                    "db_path": target_db,
                }
            )
        return out
    finally:
        conn.close()


@tool(mode="public")
def read_knowledge(source_path: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Lee detalle de una knowledge entry por `source_path` en la DB vectorial.
    """
    target_db = (db_path or "").strip() or _resolve_vector_db_path()
    target_source = (source_path or "").strip()
    if not target_source:
        return {"error": "source_path is required", "db_path": target_db}
    if not os.path.exists(target_db):
        return {"error": "DB not found", "db_path": target_db}

    conn = sqlite3.connect(target_db)
    try:
        tables = set(_hkb_table_names(target_db))
        if "docs_index" not in tables:
            return {"error": "docs_index table not found (not HKB schema)", "db_path": target_db}

        row = conn.execute(
            "SELECT source_path, n_chunks, updated_at FROM docs_index WHERE source_path = ?",
            (target_source,),
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT source_path, n_chunks, updated_at FROM docs_index WHERE source_path LIKE ? OR source_path LIKE ? LIMIT 1",
                (f"%/{target_source}", f"%\\{target_source}"),
            ).fetchone()
        if not row:
            return {"error": f"Knowledge entry not found: {target_source}", "db_path": target_db}

        source_full, n_chunks, updated_at = row

        description = ""
        ingested_at = None
        if "files_meta" in tables:
            meta_row = conn.execute(
                "SELECT COALESCE(description, ''), ingested_at FROM files_meta WHERE source_path = ?",
                (source_full,),
            ).fetchone()
            if meta_row:
                description = str(meta_row[0] or "")
                ingested_at = meta_row[1]

        page_rows = []
        chunk_preview = []
        if "chunks_meta" in tables:
            page_rows = conn.execute(
                "SELECT json_extract(locator, '$.page_start') as page, COUNT(*) as n FROM chunks_meta WHERE source_path = ? "
                "GROUP BY json_extract(locator, '$.page_start') ORDER BY json_extract(locator, '$.page_start') LIMIT 20",
                (source_full,),
            ).fetchall()
            chunk_preview = conn.execute(
                "SELECT chunk_pk, json_extract(locator, '$.page_start'), substr(json_extract(content, '$.text'), 1, 220) "
                "FROM chunks_meta WHERE source_path = ? ORDER BY rowid DESC LIMIT 5",
                (source_full,),
            ).fetchall()

        return {
            "source_path": str(source_full),
            "file": os.path.basename(str(source_full)),
            "description": description,
            "ingested_at": ingested_at,
            "n_chunks": int(n_chunks or 0),
            "updated_at": updated_at,
            "pages": [{"page": int(p), "chunks": int(n)} for p, n in page_rows],
            "chunk_preview": [
                {"chunk_id": str(cid), "page": int(pg), "md_preview": str(md or "")}
                for cid, pg, md in chunk_preview
            ],
            "db_path": target_db,
        }
    finally:
        conn.close()


def _resolve_sqlite_db_path(db_path: str) -> str:
    raw = (db_path or "").strip()
    candidates: List[str] = []

    if raw:
        candidates.append(raw)
        if raw.endswith(".db"):
            candidates.append(os.path.join(os.getcwd(), "session", raw))
            candidates.append(os.path.join(os.getcwd(), raw))
        else:
            candidates.append(os.path.join(os.getcwd(), "session", f"{raw}.db"))
            candidates.append(os.path.join(os.getcwd(), f"{raw}.db"))
        alias_map = {
            "accounting": "contabilidad.db",
            "contabilidad": "contabilidad.db",
            "transactions": "transacciones.db",
            "transacciones": "transacciones.db",
        }
        if raw.lower() in alias_map:
            alias = alias_map[raw.lower()]
            candidates.append(os.path.join(os.getcwd(), "session", alias))
            candidates.append(os.path.join(os.getcwd(), alias))
    else:
        candidates.extend(
            [
                os.path.join(os.getcwd(), "session", "contabilidad.db"),
                os.path.join(os.getcwd(), "session", "transacciones.db"),
                os.path.join(os.getcwd(), "session", "embeddings.db"),
            ]
        )

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return raw


def _sqlite_schema(db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    try:
        try:
            import sqlite_vec  # type: ignore

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            pass

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        schema: List[Dict[str, Any]] = []
        for (table_name,) in tables:
            try:
                cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            except Exception:
                cols = []
            schema.append(
                {
                    "table": str(table_name),
                    "columns": [
                        {
                            "name": str(c[1]),
                            "type": str(c[2] or ""),
                            "notnull": bool(c[3]),
                            "pk": bool(c[5]),
                        }
                        for c in cols
                    ],
                }
            )
        return schema
    finally:
        conn.close()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _guess_table(user_request: str, schema: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not schema:
        return None
    req_tokens = set(_tokenize(user_request))
    best = None
    best_score = -1
    for entry in schema:
        table_name = entry.get("table", "")
        col_names = [str(c.get("name", "")) for c in entry.get("columns", [])]
        table_tokens = set(_tokenize(table_name))
        col_tokens = set()
        for name in col_names:
            col_tokens.update(_tokenize(name))
        score = len(req_tokens.intersection(table_tokens)) * 3 + len(req_tokens.intersection(col_tokens))
        if "credito" in req_tokens and any("credito" in n.lower() for n in col_names):
            score += 2
        if score > best_score:
            best_score = score
            best = entry
    return best or schema[0]


def _looks_numeric(sql_type: str) -> bool:
    t = (sql_type or "").lower()
    return any(x in t for x in ["int", "real", "numeric", "float", "double", "decimal"])


def _preferred_select_cols(columns: List[Dict[str, Any]], max_cols: int = 6) -> List[str]:
    preferred_keywords = [
        "id",
        "credito",
        "cliente",
        "estatus",
        "status",
        "saldo",
        "monto",
        "fecha",
        "descripcion",
    ]
    names = [str(c.get("name", "")) for c in columns]
    chosen: List[str] = []
    for key in preferred_keywords:
        for name in names:
            if key in name.lower() and name not in chosen:
                chosen.append(name)
                if len(chosen) >= max_cols:
                    return chosen
    for name in names:
        if name not in chosen:
            chosen.append(name)
        if len(chosen) >= max_cols:
            break
    return chosen


def _build_where_clauses(user_request: str, columns: List[Dict[str, Any]]) -> List[str]:
    req = user_request or ""
    req_lower = req.lower()
    col_names = [str(c.get("name", "")) for c in columns]
    where: List[str] = []

    loc_match = re.search(r"\bLOC-\d{3,}\b", req, flags=re.IGNORECASE)
    if loc_match:
        loc_id = loc_match.group(0).upper()
        target_col = None
        for candidate in ["credito_id", "id_credito", "credit_id", "id"]:
            if candidate in col_names:
                target_col = candidate
                break
        if target_col is None:
            for name in col_names:
                if "credito" in name.lower() and "id" in name.lower():
                    target_col = name
                    break
        if target_col:
            where.append(f"{target_col} = '{loc_id}'")

    quoted_match = re.search(r"['\"]([^'\"]+)['\"]", req)
    if quoted_match:
        raw_text = quoted_match.group(1).strip()
        for name in col_names:
            lowered = name.lower()
            if any(key in lowered for key in ["estatus", "status", "tipo", "descripcion", "cliente"]):
                if re.search(rf"\b{re.escape(name)}\b", req_lower):
                    where.append(f"{name} = '{raw_text}'")
                    break

    return where


def _build_sql_from_request(user_request: str, table_entry: Dict[str, Any], row_limit: int) -> Dict[str, Any]:
    table_name = str(table_entry.get("table", ""))
    columns = list(table_entry.get("columns", []) or [])
    col_names = [str(c.get("name", "")) for c in columns]
    req_lower = (user_request or "").lower()

    aggregate_col = None
    for c in columns:
        if _looks_numeric(str(c.get("type", ""))):
            aggregate_col = str(c.get("name", ""))
            if any(k in aggregate_col.lower() for k in ["saldo", "monto", "total", "importe"]):
                break

    if "cuantos" in req_lower or "cuántos" in req_lower or "count" in req_lower:
        select_expr = "COUNT(*) AS total_rows"
        order_clause = ""
    elif ("promedio" in req_lower or "average" in req_lower or "avg" in req_lower) and aggregate_col:
        select_expr = f"AVG({aggregate_col}) AS avg_{aggregate_col}"
        order_clause = ""
    elif ("suma" in req_lower or "sum" in req_lower or "total" in req_lower) and aggregate_col:
        select_expr = f"SUM({aggregate_col}) AS sum_{aggregate_col}"
        order_clause = ""
    elif ("max" in req_lower or "máximo" in req_lower) and aggregate_col:
        select_expr = f"MAX({aggregate_col}) AS max_{aggregate_col}"
        order_clause = ""
    elif ("min" in req_lower or "mínimo" in req_lower) and aggregate_col:
        select_expr = f"MIN({aggregate_col}) AS min_{aggregate_col}"
        order_clause = ""
    else:
        select_cols = _preferred_select_cols(columns)
        select_expr = ", ".join(select_cols) if select_cols else "*"
        if "fecha" in col_names:
            order_clause = " ORDER BY fecha DESC"
        elif "id" in col_names:
            order_clause = " ORDER BY id DESC"
        else:
            order_clause = ""

    where_clauses = _build_where_clauses(user_request, columns)
    where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"SELECT {select_expr} FROM {table_name}{where_sql}{order_clause} LIMIT {max(1, min(row_limit, 500))}"
    return {
        "sql": sql,
        "table": table_name,
        "select_expr": select_expr,
        "where_clauses": where_clauses,
    }


def _run_readonly_sql(db_path: str, sql: str) -> Dict[str, Any]:
    lowered = (sql or "").strip().lower()
    if not lowered.startswith("select"):
        return {"ok": False, "error": "Only SELECT queries are allowed."}
    forbidden = (" insert ", " update ", " delete ", " drop ", " alter ", " create ", " pragma ")
    padded = f" {lowered} "
    if any(tok in padded for tok in forbidden):
        return {"ok": False, "error": "Query contains forbidden SQL operations."}

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in (cur.description or [])]
        return {"ok": True, "columns": cols, "rows": rows, "row_count": len(rows)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        conn.close()


@tool(mode="public")
def nl2sql_sqlite(
    user_request: str,
    db_path: str = "",
    row_limit: int = 50,
    execute: bool = False,
) -> Dict[str, Any]:
    """
    Genera SQL SELECT desde lenguaje natural usando el schema real de una SQLite.
    Opcionalmente ejecuta la consulta generada en modo read-only.
    """
    target_db = _resolve_sqlite_db_path(db_path)
    if not target_db or not os.path.exists(target_db):
        return {
            "ok": False,
            "error": f"SQLite DB not found: {target_db or db_path}",
            "db_path": target_db or db_path,
        }

    try:
        schema = _sqlite_schema(target_db)
    except Exception as exc:
        return {"ok": False, "error": f"Failed to inspect schema: {exc}", "db_path": target_db}

    if not schema:
        return {"ok": False, "error": "No user tables found in DB.", "db_path": target_db}

    chosen_table = _guess_table(user_request, schema)
    if not chosen_table:
        return {"ok": False, "error": "Could not infer target table.", "db_path": target_db}

    plan = _build_sql_from_request(user_request, chosen_table, row_limit=row_limit)
    result: Dict[str, Any] = {
        "ok": True,
        "db_path": target_db,
        "user_request": user_request,
        "generated_sql": plan["sql"],
        "chosen_table": plan["table"],
        "where_clauses": plan["where_clauses"],
        "schema": schema,
    }

    if execute:
        exec_out = _run_readonly_sql(target_db, plan["sql"])
        result["execution"] = exec_out
    return result


class _NL2SQLToolAgentSQLite:
    """
    Tool-agent: encapsulates a mini pipeline so the EXECUTOR only needs one tool_call.

    Today it uses the same schema-aware heuristic as nl2sql_sqlite().
    Later we can swap the planner step with an LLM backend while keeping the JSON contract stable.
    """

    def run(
        self,
        user_request: str,
        db_path: str,
        row_limit: int,
        execute: bool,
    ) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []

        target_db = _resolve_sqlite_db_path(db_path)
        trace.append({"step": "resolve_db", "input": db_path, "output": target_db})
        if not target_db or not os.path.exists(target_db):
            return {
                "ok": False,
                "error": f"SQLite DB not found: {target_db or db_path}",
                "trace": trace,
                "db_path": target_db or db_path,
            }

        try:
            schema = _sqlite_schema(target_db)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Failed to inspect schema: {exc}",
                "trace": trace,
                "db_path": target_db,
            }

        trace.append({"step": "inspect_schema", "tables": [t.get("table") for t in schema]})
        if not schema:
            return {
                "ok": False,
                "error": "No user tables found in DB.",
                "trace": trace,
                "db_path": target_db,
            }

        chosen_table = _guess_table(user_request, schema)
        if not chosen_table:
            return {
                "ok": False,
                "error": "Could not infer target table.",
                "trace": trace,
                "db_path": target_db,
                "schema": schema,
            }

        trace.append({"step": "choose_table", "table": chosen_table.get("table")})
        plan = _build_sql_from_request(user_request, chosen_table, row_limit=row_limit)
        trace.append(
            {
                "step": "generate_sql",
                "sql": plan.get("sql"),
                "where_clauses": plan.get("where_clauses", []),
            }
        )

        out: Dict[str, Any] = {
            "ok": True,
            "agent": "nl2sql_agent_sqlite",
            "db_path": target_db,
            "user_request": user_request,
            "chosen_table": plan.get("table"),
            "generated_sql": plan.get("sql"),
            "where_clauses": plan.get("where_clauses", []),
            "trace": trace,
        }

        if execute:
            exec_out = _run_readonly_sql(target_db, plan.get("sql", ""))
            out["execution"] = exec_out
            trace.append({"step": "execute_sql", "ok": bool(exec_out.get("ok"))})
        return out


@tool(mode="public")
def nl2sql_agent_sqlite(
    user_request: str,
    db_path: str = "",
    row_limit: int = 50,
    execute: bool = True,
) -> Dict[str, Any]:
    """
    NL2SQL como agente independiente (tool-agent).

    El EXECUTOR solo necesita invocar UNA tool: esta tool inspecciona el schema,
    genera SQL SELECT read-only y (opcionalmente) lo ejecuta, devolviendo un JSON con traza.
    """
    agent = _NL2SQLToolAgentSQLite()
    return agent.run(
        user_request=user_request,
        db_path=db_path,
        row_limit=row_limit,
        execute=execute,
    )


def _extract_first_json_obj(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    frag = raw[start : end + 1]
    try:
        parsed = json.loads(frag)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _ensure_select_limit(sql: str, row_limit: int) -> str:
    txt = (sql or "").strip().rstrip(";")
    if not txt:
        return ""
    low = txt.lower()
    if not low.startswith("select"):
        return ""
    forbidden = (" insert ", " update ", " delete ", " drop ", " alter ", " create ", " pragma ")
    padded = f" {low} "
    if any(tok in padded for tok in forbidden):
        return ""
    if " limit " not in low:
        txt = f"{txt} LIMIT {max(1, min(int(row_limit or 50), 500))}"
    return txt


def _schema_compact_text(schema: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for table in schema:
        tname = str(table.get("table", ""))
        cols = table.get("columns", []) or []
        col_bits = []
        for c in cols:
            col_bits.append(f"{c.get('name')}:{c.get('type')}")
        lines.append(f"{tname}({', '.join(col_bits)})")
    return "\n".join(lines)


def _nl2sql_generate_with_provider(
    user_request: str,
    schema: List[Dict[str, Any]],
    row_limit: int,
) -> Dict[str, Any]:
    from agnostic_agent.providers.factory import ProviderFactory

    provider = os.getenv("KNOWLEDGE_VOYAGUE_NL2SQL_LLM_PROVIDER", "").strip().lower()
    if not provider:
        if os.getenv("BEDROCK_MODEL_ID", "").strip() or os.getenv("KNOWLEDGE_VOYAGUE_BEDROCK_MODEL", "").strip():
            provider = "bedrock"
        elif os.getenv("VLLM_API_BASE", "").strip():
            provider = "vllm"
        else:
            provider = "openai"

    if provider == "bedrock":
        llm_cfg: Dict[str, Any] = {
            "provider": "bedrock",
            "model": os.getenv("KNOWLEDGE_VOYAGUE_BEDROCK_MODEL", "").strip()
            or os.getenv("BEDROCK_MODEL_ID", "").strip()
            or "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "region_name": os.getenv("BEDROCK_REGION", "").strip() or os.getenv("AWS_REGION", "").strip(),
            "api": os.getenv("BEDROCK_API_MODE", "").strip() or "auto",
        }
    elif provider == "vllm":
        llm_cfg = {
            "provider": "vllm",
            "model": os.getenv("KNOWLEDGE_VOYAGUE_NL2SQL_MODEL", "").strip()
            or os.getenv("LLM_SERVED_NAME", "").strip()
            or "agnostic-llm",
            "base_url": os.getenv("VLLM_API_BASE", "").strip() or "http://127.0.0.1:8000/v1",
            "api_key": os.getenv("OPENAI_API_KEY", "").strip() or "EMPTY",
        }
    else:
        llm_cfg = {
            "provider": "openai",
            "model": os.getenv("KNOWLEDGE_VOYAGUE_NL2SQL_MODEL", "").strip()
            or os.getenv("OPENAI_MODEL", "").strip()
            or "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        }

    system_prompt = (
        "You are an NL2SQL planner for SQLite.\n"
        "Return ONLY strict JSON: {\"sql\": \"...\", \"reason\": \"...\"}\n"
        "Rules:\n"
        "- Use ONLY tables/columns from the provided schema.\n"
        "- Generate a single SELECT statement.\n"
        "- No DDL/DML/PRAGMA.\n"
        f"- Always include LIMIT <= {max(1, min(int(row_limit or 50), 500))}.\n"
        "- If uncertain, still return the safest best-effort SELECT over likely table/columns.\n"
    )
    user_prompt = (
        "Schema:\n"
        f"{_schema_compact_text(schema)}\n\n"
        "User request:\n"
        f"{user_request}\n"
    )

    llm = ProviderFactory.get_llm_provider(llm_cfg)
    text = llm.generate(user_prompt, system_prompt=system_prompt, temperature=0.0)
    parsed = _extract_first_json_obj(text)
    sql = _ensure_select_limit(str(parsed.get("sql", "")), row_limit=row_limit)
    return {
        "provider": provider,
        "model": llm_cfg.get("model"),
        "raw": text,
        "sql": sql,
        "reason": str(parsed.get("reason", "")),
    }


@tool(mode="public")
def knowledge_voyague_nl2sql_agent(
    user_request: str,
    db_path: str = "",
    row_limit: int = 50,
    execute: bool = True,
) -> Dict[str, Any]:
    """
    Tool-agent NL2SQL for centralized knowledge DBs.
    Pipeline: resolve db -> inspect schema -> LLM SQL generation -> read-only execution.
    Returns a full JSON trace for catcher/dev.
    """
    target_db = _resolve_sqlite_db_path(db_path) if db_path else _resolve_vector_db_path()
    trace: List[Dict[str, Any]] = [{"step": "resolve_db", "db_path": target_db}]
    if not target_db or not os.path.exists(target_db):
        return {"ok": False, "error": f"DB not found: {target_db}", "trace": trace}

    try:
        schema = _sqlite_schema(target_db)
    except Exception as exc:
        return {"ok": False, "error": f"Schema read failed: {exc}", "trace": trace, "db_path": target_db}

    trace.append({"step": "inspect_schema", "tables": [t.get("table") for t in schema]})
    if not schema:
        return {"ok": False, "error": "No user tables in target DB.", "trace": trace, "db_path": target_db}

    sql = ""
    gen_meta: Dict[str, Any] = {}
    try:
        gen_meta = _nl2sql_generate_with_provider(user_request, schema, row_limit=row_limit)
        sql = gen_meta.get("sql", "") or ""
        trace.append(
            {
                "step": "llm_generate_sql",
                "provider": gen_meta.get("provider"),
                "model": gen_meta.get("model"),
                "has_sql": bool(sql),
            }
        )
    except Exception as exc:
        trace.append({"step": "llm_generate_sql", "error": str(exc)})

    if not sql:
        chosen = _guess_table(user_request, schema)
        if not chosen:
            return {
                "ok": False,
                "error": "Could not infer SQL from LLM and fallback heuristic failed.",
                "trace": trace,
                "db_path": target_db,
            }
        plan = _build_sql_from_request(user_request, chosen, row_limit=row_limit)
        sql = _ensure_select_limit(plan.get("sql", ""), row_limit=row_limit)
        trace.append({"step": "heuristic_fallback_sql", "table": chosen.get("table"), "sql": sql})

    if not sql:
        return {
            "ok": False,
            "error": "Unsafe or empty SQL generated.",
            "trace": trace,
            "db_path": target_db,
            "llm": {"provider": gen_meta.get("provider"), "model": gen_meta.get("model")},
        }

    out: Dict[str, Any] = {
        "ok": True,
        "agent": "knowledge_voyague_nl2sql_agent",
        "db_path": target_db,
        "user_request": user_request,
        "generated_sql": sql,
        "trace": trace,
        "llm": {
            "provider": gen_meta.get("provider"),
            "model": gen_meta.get("model"),
            "reason": gen_meta.get("reason", ""),
        },
    }
    if execute:
        exec_out = _run_readonly_sql(target_db, sql)
        out["execution"] = exec_out
        trace.append({"step": "execute_sql", "ok": bool(exec_out.get("ok"))})
    return out


@tool(mode="public")
def knowledge_voyague_nl2semantic_agent(
    query: str,
    top_k: int = 15,
    rerank_top_n: int = 5,
    source_filter: str = "",
) -> Dict[str, Any]:
    """
    Tool-agent NL2SEMANTIC over centralized embeddings DB.
    Pipeline: semantic retrieval top_k + reranker top_n.
    Returns JSON trace and candidates.
    """
    from agnostic_agent.tools.semantic import search_knowledge_base, rerank_docs
    from agnostic_agent.providers.factory import ProviderFactory

    def _choose_llm_provider_for_classifier() -> Dict[str, Any]:
        provider = os.getenv("KNOWLEDGE_VOYAGUE_CLASSIFIER_PROVIDER", "").strip().lower()
        if not provider:
            if os.getenv("BEDROCK_MODEL_ID", "").strip() or os.getenv("KNOWLEDGE_VOYAGUE_BEDROCK_MODEL", "").strip():
                provider = "bedrock"
            elif os.getenv("VLLM_API_BASE", "").strip():
                provider = "vllm"
            else:
                provider = "openai"

        if provider == "bedrock":
            return {
                "provider": "bedrock",
                "model": os.getenv("KNOWLEDGE_VOYAGUE_BEDROCK_MODEL", "").strip()
                or os.getenv("BEDROCK_MODEL_ID", "").strip()
                or "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "region_name": os.getenv("BEDROCK_REGION", "").strip() or os.getenv("AWS_REGION", "").strip(),
                "api": os.getenv("BEDROCK_API_MODE", "").strip() or "auto",
            }
        if provider == "vllm":
            return {
                "provider": "vllm",
                "model": os.getenv("KNOWLEDGE_VOYAGUE_CLASSIFIER_MODEL", "").strip()
                or os.getenv("LLM_SERVED_NAME", "").strip()
                or "agnostic-llm",
                "base_url": os.getenv("VLLM_API_BASE", "").strip() or "http://127.0.0.1:8000/v1",
                "api_key": os.getenv("OPENAI_API_KEY", "").strip() or "EMPTY",
            }
        return {
            "provider": "openai",
            "model": os.getenv("KNOWLEDGE_VOYAGUE_CLASSIFIER_MODEL", "").strip()
            or os.getenv("OPENAI_MODEL", "").strip()
            or "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        }

    def _llm_rerank(hits_local: List[Dict[str, Any]], top_n: int) -> Dict[str, Any]:
        llm_cfg = _choose_llm_provider_for_classifier()
        llm = ProviderFactory.get_llm_provider(llm_cfg)
        ranked: List[Dict[str, Any]] = []
        system_prompt = (
            "You are a strict relevance classifier.\n"
            "Given query and one candidate chunk, score relevance from 0.0 to 1.0.\n"
            "Return JSON only: {\"score\": <float>, \"reason\": \"<short>\"}.\n"
        )
        for row in hits_local:
            chunk_text = str(row.get("md") or "")[:2000]
            user_prompt = (
                f"Query:\n{query}\n\n"
                f"Candidate:\n{chunk_text}\n"
            )
            raw = llm.generate(user_prompt, system_prompt=system_prompt, temperature=0.0)
            parsed = _extract_first_json_obj(raw)
            score = parsed.get("score", row.get("score", 0.0))
            try:
                score_f = float(score)
            except Exception:
                score_f = float(row.get("score", 0.0) or 0.0)
            ranked.append({"score": score_f, "document": row, "reason": str(parsed.get("reason", ""))})

        ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return {
            "provider": llm_cfg.get("provider"),
            "model": llm_cfg.get("model"),
            "ranked": ranked[: max(1, int(top_n or 5))],
        }

    trace: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {"query": query, "top_k": int(top_k or 15)}
    if (source_filter or "").strip():
        payload["source_filter"] = source_filter.strip()
        payload["source_filter_origin"] = "planner"

    raw_hits = search_knowledge_base.invoke(payload)
    trace.append({"step": "semantic_retrieve", "requested_top_k": int(top_k or 15), "raw_count": len(raw_hits or [])})

    hits: List[Dict[str, Any]] = []
    for item in raw_hits or []:
        if not isinstance(item, dict):
            continue
        if item.get("_meta_only"):
            continue
        if item.get("error"):
            return {"ok": False, "agent": "knowledge_voyague_nl2semantic_agent", "error": item.get("error"), "trace": trace}
        hits.append(item)

    rerank_strategy = (os.getenv("KNOWLEDGE_VOYAGUE_RERANK_STRATEGY", "auto") or "auto").strip().lower()
    if rerank_strategy not in ("auto", "model", "llm"):
        rerank_strategy = "auto"

    use_llm_classifier = rerank_strategy == "llm"
    if rerank_strategy == "auto":
        openai_mode = bool(os.getenv("OPENAI_MODEL", "").strip())
        has_reranker_model = bool(os.getenv("RERANK_MODEL_ID", "").strip())
        use_llm_classifier = openai_mode and not has_reranker_model

    reranked = []
    classifier_meta: Dict[str, Any] = {}
    if use_llm_classifier:
        try:
            rr = _llm_rerank(hits, int(rerank_top_n or 5))
            reranked = rr.get("ranked", [])
            classifier_meta = {"provider": rr.get("provider"), "model": rr.get("model"), "strategy": "llm_classifier"}
        except Exception as exc:
            trace.append({"step": "llm_classifier_error", "error": str(exc)})
            ordered = sorted(hits, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            reranked = [{"score": float(h.get("score", 0.0) or 0.0), "document": h, "reason": "semantic_score_fallback"} for h in ordered[: max(1, int(rerank_top_n or 5))]]
            classifier_meta = {"strategy": "semantic_score_fallback"}
    else:
        try:
            reranked = rerank_docs.invoke(
                {
                    "query": query,
                    "documents": hits,
                    "top_n": int(rerank_top_n or 5),
                }
            )
            classifier_meta = {"strategy": "reranker_model"}
        except Exception as exc:
            trace.append({"step": "reranker_model_error", "error": str(exc)})
            try:
                rr = _llm_rerank(hits, int(rerank_top_n or 5))
                reranked = rr.get("ranked", [])
                classifier_meta = {"provider": rr.get("provider"), "model": rr.get("model"), "strategy": "llm_classifier_fallback"}
            except Exception as exc2:
                trace.append({"step": "llm_fallback_error", "error": str(exc2)})
                ordered = sorted(hits, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
                reranked = [{"score": float(h.get("score", 0.0) or 0.0), "document": h, "reason": "semantic_score_fallback"} for h in ordered[: max(1, int(rerank_top_n or 5))]]
                classifier_meta = {"strategy": "semantic_score_fallback"}

    trace.append(
        {
            "step": "rerank",
            "strategy": classifier_meta.get("strategy"),
            "input_count": len(hits),
            "output_count": len(reranked or []),
        }
    )

    results: List[Dict[str, Any]] = []
    for item in reranked or []:
        if not isinstance(item, dict):
            continue
        doc = item.get("document")
        if isinstance(doc, dict):
            results.append(
                {
                    "rank_score": float(item.get("score", 0.0)),
                    "source_path": doc.get("source_path"),
                    "page": doc.get("page"),
                    "chunk_id": doc.get("chunk_id"),
                    "semantic_score": doc.get("score"),
                    "md": doc.get("md"),
                }
            )
        else:
            results.append({"rank_score": float(item.get("score", 0.0)), "document": doc})

    return {
        "ok": True,
        "agent": "knowledge_voyague_nl2semantic_agent",
        "query": query,
        "top_k": int(top_k or 15),
        "rerank_top_n": int(rerank_top_n or 5),
        "results": results,
        "classifier": classifier_meta,
        "trace": trace,
    }
