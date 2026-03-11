from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Tuple


def execute_sql(db_path: str, sql: str) -> Dict[str, Any]:
    lowered = (sql or "").strip().lower()
    if not lowered.startswith("select") and not lowered.startswith("with"):
        return {"ok": False, "error": "Only SELECT/WITH queries are allowed."}
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


def summarize_result(user_query: str, execution: Dict[str, Any]) -> str:
    if not execution.get("ok"):
        return f"No se pudo ejecutar la consulta para responder '{user_query}'. Error: {execution.get('error', '')}"
    rows = execution.get("rows") or []
    if not rows:
        return f"No se encontraron filas para responder: {user_query}."
    if len(rows) == 1 and len(rows[0]) == 1:
        return f"Resultado: {rows[0][0]!r} para la consulta: {user_query}."
    if len(rows) == 1:
        return f"Se obtuvo 1 fila con {len(rows[0])} columnas."
    return f"Se obtuvieron {len(rows)} filas y {len(rows[0]) if rows else 0} columnas."
