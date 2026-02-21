from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import sys
import sqlite3

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
                "SELECT page, COUNT(*) as n FROM chunks_meta WHERE source_path = ? GROUP BY page ORDER BY page LIMIT 20",
                (source_full,),
            ).fetchall()
            chunk_preview = conn.execute(
                "SELECT chunk_id, page, substr(md, 1, 220) FROM chunks_meta WHERE source_path = ? ORDER BY rowid DESC LIMIT 5",
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
