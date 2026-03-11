from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _table_names(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(r[0]) for r in rows]


def _column_dict(row: Iterable[Any]) -> Dict[str, Any]:
    values = list(row)
    return {
        "name": str(values[1]),
        "type": str(values[2] or ""),
        "nullable": not bool(values[3]),
        "description": "",
        "examples": [],
        "constraints": ["PRIMARY KEY"] if bool(values[5]) else [],
    }


def _foreign_keys(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    rows = conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "columns": [str(row[3])],
                "ref_table": str(row[2]),
                "ref_columns": [str(row[4])],
            }
        )
    return out


def _indexes(conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
    rows = conn.execute(f"PRAGMA index_list({table_name})").fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        index_name = str(row[1])
        index_cols = conn.execute(f"PRAGMA index_info({index_name})").fetchall()
        out.append(
            {
                "name": index_name,
                "columns": [str(col[2]) for col in index_cols],
                "unique": bool(row[2]),
            }
        )
    return out


def build_sqlite_catalog(
    *,
    db_path: str,
    output_path: str,
    descriptions: Optional[Dict[str, str]] = None,
    business_glossary: Optional[Dict[str, str]] = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    descriptions = descriptions or {}
    business_glossary = business_glossary or {}
    defaults = defaults or {}

    source = Path(db_path)
    if not source.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    conn = sqlite3.connect(str(source))
    try:
        tables: Dict[str, Any] = {}
        relationships: List[Dict[str, Any]] = []
        for table_name in _table_names(conn):
            col_rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            cols = [_column_dict(row) for row in col_rows]
            fks = _foreign_keys(conn, table_name)
            pk_cols = [str(row[1]) for row in col_rows if bool(row[5])]
            idxs = _indexes(conn, table_name)
            tables[table_name] = {
                "description": descriptions.get(table_name, ""),
                "primary_key": pk_cols,
                "columns": cols,
                "foreign_keys": fks,
                "indexes": idxs,
                "row_count_hint": None,
            }
            for fk in fks:
                relationships.append(
                    {
                        "left_table": table_name,
                        "left_column": fk["columns"][0] if fk["columns"] else "",
                        "right_table": fk["ref_table"],
                        "right_column": fk["ref_columns"][0] if fk["ref_columns"] else "",
                        "cardinality": "N:1",
                        "description": "",
                    }
                )

        catalog = {
            "version": "2.0",
            "engine": "sqlite",
            "source_db": str(source),
            "allowed_schemas": ["main"],
            "defaults": defaults,
            "business_glossary": business_glossary,
            "schemas": {"main": {"tables": tables, "views": {}}},
            "relationships": relationships,
            "table_summaries": {table: meta.get("description", "") for table, meta in tables.items()},
        }
    finally:
        conn.close()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog
