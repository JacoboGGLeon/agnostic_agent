from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from agnostic_agent.tools.decorators import tool


def _default_finance_dir() -> Path:
    # Repo layout expected:
    # AVANTECK.TEAM/
    #   - agnostic_agent/
    #   - ais/examples/finance/
    workspace_root = Path(__file__).resolve().parents[3]
    return workspace_root / "ais" / "examples" / "finance"


def _transactions_db_path() -> Path:
    import os

    return Path(
        os.getenv(
            "AGNOSTIC_FIN_TRANS_DB",
            str(_default_finance_dir() / "transacciones.db"),
        )
    )


def _accounting_db_path() -> Path:
    import os

    return Path(
        os.getenv(
            "AGNOSTIC_FIN_ACC_DB",
            str(_default_finance_dir() / "contabilidad.db"),
        )
    )


def _is_read_only_sql(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    # Keep tool strictly read-only.
    if not q.startswith("select"):
        return False
    forbidden = ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "pragma ")
    return not any(tok in q for tok in forbidden)


def _run_query(db_path: Path, query: str) -> str:
    if not _is_read_only_sql(query):
        return "Error SQL: solo se permiten consultas SELECT de solo lectura."
    if not db_path.exists():
        return f"Error SQL: no se encontro la base de datos: {db_path}"

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [d[0] for d in (cur.description or [])]
        conn.close()
        payload = {"columns": columns, "rows": rows}
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return f"Error SQL: {exc}"


@tool(mode="public")
def query_transactions_db(query: str) -> str:
    """
    Ejecuta una consulta SELECT de solo lectura sobre transacciones (Universo 1).
    Devuelve JSON string con `columns` y `rows`.
    """
    return _run_query(_transactions_db_path(), query)


@tool(mode="public")
def query_accounting_db(query: str) -> str:
    """
    Ejecuta una consulta SELECT de solo lectura sobre contabilidad (Universo 2).
    Devuelve JSON string con `columns` y `rows`.
    """
    return _run_query(_accounting_db_path(), query)


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = text.replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text


_SANEAMIENTO_RATES: Dict[str, float] = {
    "desembolsado": 0.01,
    "vigente / al corriente": 0.01,
    "mora temprana (1-30 dias)": 0.05,
    "mora media (31-60 dias)": 0.20,
    "mora tardia (61-90 dias)": 0.50,
    "cartera vencida (+90 dias)": 1.00,
    "castigado / incobrable": 1.00,
    "en cobranza externa / legal": 1.00,
    "liquidado / cerrado": 0.00,
}


@tool(mode="public")
def get_saneamiento_rate(estatus: str) -> Dict[str, Any]:
    """
    Devuelve la tasa de saneamiento esperada para un estatus crediticio.
    """
    key = _normalize_text(estatus)
    rate = _SANEAMIENTO_RATES.get(key)
    if rate is None:
        return {
            "found": False,
            "estatus": estatus,
            "estatus_normalized": key,
            "known_statuses": sorted(_SANEAMIENTO_RATES.keys()),
        }
    return {
        "found": True,
        "estatus": estatus,
        "estatus_normalized": key,
        "tasa_saneamiento": rate,
    }


def _fetch_transactions(credito_id: str) -> List[tuple[str, float]]:
    db = sqlite3.connect(str(_transactions_db_path()))
    cur = db.cursor()
    cur.execute(
        "SELECT tipo, monto FROM movimientos WHERE credito_id = ?",
        (credito_id,),
    )
    rows = cur.fetchall()
    db.close()
    return [(str(tipo), float(monto)) for tipo, monto in rows]


def _fetch_accounting(credito_id: str) -> tuple[float, str, float]:
    db = sqlite3.connect(str(_accounting_db_path()))
    cur = db.cursor()
    cur.execute(
        """
        SELECT saldo_total, estatus, saneamiento_calculado
        FROM estados_cuenta
        WHERE credito_id = ?
        """,
        (credito_id,),
    )
    row = cur.fetchone()
    db.close()
    if row is None:
        raise ValueError(f"No existe credito_id={credito_id} en contabilidad.")
    return float(row[0]), str(row[1]), float(row[2])


@tool(mode="public")
def reconcile_credit_accounting(credito_id: str) -> Dict[str, Any]:
    """
    Concilia un credito de forma determinista en modo 1-a-1:
    1) Flujos (desembolsos/pagos/penalizaciones/descuentos)
    2) Estado contable (saldo_total, estatus, saneamiento_calculado)
    3) Validacion de saldo
    4) Validacion de saneamiento
    """
    try:
        tx_rows = _fetch_transactions(credito_id)
        saldo_total, estatus, saneamiento_calculado = _fetch_accounting(credito_id)
    except Exception as exc:
        return {"ok": False, "credito_id": credito_id, "error": str(exc)}

    totals = {
        "DESEMBOLSO": 0.0,
        "PAGO": 0.0,
        "PENALIZACION": 0.0,
        "DESCUENTO": 0.0,
    }
    for tipo, monto in tx_rows:
        key = str(tipo).upper().strip()
        if key in totals:
            totals[key] += float(monto)

    saldo_esperado = (
        (totals["DESEMBOLSO"] - totals["PAGO"])
        + totals["PENALIZACION"]
        - totals["DESCUENTO"]
    )
    diff_saldo = round(saldo_total - saldo_esperado, 2)
    saldo_ok = abs(diff_saldo) < 0.01

    rate_info = get_saneamiento_rate.invoke({"estatus": estatus})
    if isinstance(rate_info, dict) and rate_info.get("found"):
        tasa = float(rate_info["tasa_saneamiento"])
    else:
        tasa = 0.0
    reserva_esperada = round(saldo_total * tasa, 2)
    diff_reserva = round(saneamiento_calculado - reserva_esperada, 2)
    saneamiento_ok = abs(diff_reserva) < 0.01

    status = "CUADRADO (100% Match)" if saldo_ok and saneamiento_ok else "DRIFT DETECTADO"
    return {
        "ok": True,
        "credito_id": credito_id,
        "estatus": estatus,
        "status": status,
        "validaciones": {
            "saldo_ok": saldo_ok,
            "saneamiento_ok": saneamiento_ok,
        },
        "flujos": totals,
        "saldo": {
            "reportado": round(saldo_total, 2),
            "esperado": round(saldo_esperado, 2),
            "diferencia": diff_saldo,
        },
        "saneamiento": {
            "tasa": tasa,
            "reportado": round(saneamiento_calculado, 2),
            "esperado": reserva_esperada,
            "diferencia": diff_reserva,
        },
    }
