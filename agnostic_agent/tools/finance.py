from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from agnostic_agent.tools.decorators import tool


def _default_finance_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[3]
    return workspace_root / "ais" / "examples" / "finance"


def _resolve_path(env_var: str, default_path: Path, fallback_candidates: List[Path]) -> Path:
    env_value = os.getenv(env_var, "").strip()
    if env_value:
        return Path(env_value)
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate
    return default_path


def _transactions_db_path() -> Path:
    default_path = _default_finance_dir() / "transacciones.db"
    return _resolve_path(
        "AGNOSTIC_FIN_TRANS_DB",
        default_path,
        [
            Path("/content/session/transacciones.db"),
            Path.cwd() / "session" / "transacciones.db",
            Path.cwd() / "transacciones.db",
            default_path,
        ],
    )


def _accounting_db_path() -> Path:
    default_path = _default_finance_dir() / "contabilidad.db"
    return _resolve_path(
        "AGNOSTIC_FIN_ACC_DB",
        default_path,
        [
            Path("/content/session/contabilidad.db"),
            Path.cwd() / "session" / "contabilidad.db",
            Path.cwd() / "contabilidad.db",
            default_path,
        ],
    )


def _rules_md_path() -> Path:
    default_path = _default_finance_dir() / "knowledge" / "rules.md"
    return _resolve_path(
        "AGNOSTIC_FIN_RULES_MD",
        default_path,
        [
            Path("/content/session/rules.md"),
            Path.cwd() / "session" / "rules.md",
            Path.cwd() / "rules.md",
            default_path,
        ],
    )


def _dictionary_md_path() -> Path:
    default_path = _default_finance_dir() / "knowledge" / "dictionary.md"
    return _resolve_path(
        "AGNOSTIC_FIN_DICT_MD",
        default_path,
        [
            Path("/content/session/dictionary.md"),
            Path.cwd() / "session" / "dictionary.md",
            Path.cwd() / "dictionary.md",
            default_path,
        ],
    )


def _is_read_only_sql(query: str) -> bool:
    lowered = (query or "").strip().lower()
    if not lowered:
        return False
    if not lowered.startswith("select"):
        return False
    forbidden = ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "pragma ")
    return not any(token in lowered for token in forbidden)


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
        columns = [desc[0] for desc in (cur.description or [])]
        conn.close()
        return json.dumps({"columns": columns, "rows": rows}, ensure_ascii=False)
    except Exception as exc:
        return f"Error SQL: {exc}"


@tool(mode="public")
def query_transactions_db(query: str) -> str:
    """
    Ejecuta consulta SELECT de solo lectura sobre transacciones.
    """
    return _run_query(_transactions_db_path(), query)


@tool(mode="public")
def query_accounting_db(query: str) -> str:
    """
    Ejecuta consulta SELECT de solo lectura sobre contabilidad.
    """
    return _run_query(_accounting_db_path(), query)


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("â€“", "-").replace("â€”", "-")
    text = re.sub(r"\s+", " ", text)
    return text


_SANEAMIENTO_RATES_DEFAULT: Dict[str, float] = {
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

_RULES_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "rates": None}


def _strict_rules_mode() -> bool:
    return os.getenv("AGNOSTIC_FIN_STRICT_RULES", "0").lower() in ("1", "true", "yes", "on")


def _parse_percent_to_rate(raw: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", raw or "")
    if not match:
        return None
    return float(match.group(1)) / 100.0


def _load_rates_from_rules_md(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8", errors="ignore")
    rates: Dict[str, float] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if "Estatus" in line:
            continue
        if re.fullmatch(r"\|[-\s|:]+\|?", line):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 3:
            continue
        rate = _parse_percent_to_rate(parts[2])
        if rate is None:
            continue
        rates[_normalize_text(parts[0])] = rate
    return rates


def _get_runtime_rates() -> Dict[str, float]:
    rules_path = _rules_md_path()
    mtime = rules_path.stat().st_mtime if rules_path.exists() else None
    if (
        _RULES_CACHE.get("path") == str(rules_path)
        and _RULES_CACHE.get("mtime") == mtime
        and isinstance(_RULES_CACHE.get("rates"), dict)
    ):
        return _RULES_CACHE["rates"]
    parsed = _load_rates_from_rules_md(rules_path)
    if not parsed and not _strict_rules_mode():
        parsed = dict(_SANEAMIENTO_RATES_DEFAULT)
    _RULES_CACHE["path"] = str(rules_path)
    _RULES_CACHE["mtime"] = mtime
    _RULES_CACHE["rates"] = parsed
    return parsed


@tool(mode="public")
def finance_sources_status() -> Dict[str, Any]:
    """
    Reporta estado de fuentes financieras (DBs y markdown de reglas/diccionario).
    """
    paths = {
        "transactions_db": _transactions_db_path(),
        "accounting_db": _accounting_db_path(),
        "rules_md": _rules_md_path(),
        "dictionary_md": _dictionary_md_path(),
    }
    rates = _get_runtime_rates()
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "exists": {name: path.exists() for name, path in paths.items()},
        "rules_loaded_count": len(rates),
        "rules_source": "rules.md" if _rules_md_path().exists() and rates else "fallback_default",
        "strict_rules_mode": _strict_rules_mode(),
    }


@tool(mode="public")
def get_saneamiento_rate(estatus: str) -> Dict[str, Any]:
    """
    Devuelve la tasa de saneamiento esperada para un estatus crediticio.
    """
    rates = _get_runtime_rates()
    key = _normalize_text(estatus)
    rate = rates.get(key)
    if rate is None:
        return {
            "found": False,
            "estatus": estatus,
            "estatus_normalized": key,
            "known_statuses": sorted(rates.keys()),
            "rules_path": str(_rules_md_path()),
            "dictionary_path": str(_dictionary_md_path()),
            "strict_rules_mode": _strict_rules_mode(),
        }
    return {
        "found": True,
        "estatus": estatus,
        "estatus_normalized": key,
        "tasa_saneamiento": rate,
        "rules_path": str(_rules_md_path()),
        "dictionary_path": str(_dictionary_md_path()),
        "strict_rules_mode": _strict_rules_mode(),
    }


def _fetch_transactions(credito_id: str) -> List[tuple[str, float]]:
    db = sqlite3.connect(str(_transactions_db_path()))
    cur = db.cursor()
    cur.execute("SELECT tipo, monto FROM movimientos WHERE credito_id = ?", (credito_id,))
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
def reconcile_credit_accounting(credito_id: str, balance: str = "") -> Dict[str, Any]:
    """
    Concilia un credito 1-a-1 y valida saldo y saneamiento.
    """
    _ = balance
    try:
        tx_rows = _fetch_transactions(credito_id)
        saldo_total, estatus, saneamiento_calculado = _fetch_accounting(credito_id)
    except Exception as exc:
        return {"ok": False, "credito_id": credito_id, "error": str(exc)}

    totals = {"DESEMBOLSO": 0.0, "PAGO": 0.0, "PENALIZACION": 0.0, "DESCUENTO": 0.0}
    for tipo, monto in tx_rows:
        key = str(tipo).upper().strip()
        if key in totals:
            totals[key] += float(monto)

    saldo_esperado = (totals["DESEMBOLSO"] - totals["PAGO"]) + totals["PENALIZACION"] - totals["DESCUENTO"]
    diff_saldo = round(saldo_total - saldo_esperado, 2)
    saldo_ok = abs(diff_saldo) < 0.01

    rate_info = get_saneamiento_rate.invoke({"estatus": estatus})
    if isinstance(rate_info, dict) and rate_info.get("found"):
        tasa = float(rate_info["tasa_saneamiento"])
    else:
        if _strict_rules_mode():
            return {
                "ok": False,
                "credito_id": credito_id,
                "error": (
                    "No fue posible resolver la tasa de saneamiento desde rules.md "
                    f"para estatus='{estatus}' en modo estricto."
                ),
                "sources": {
                    "rules_path": str(_rules_md_path()),
                    "dictionary_path": str(_dictionary_md_path()),
                },
            }
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
        "validaciones": {"saldo_ok": saldo_ok, "saneamiento_ok": saneamiento_ok},
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
        "sources": {
            "rules_path": str(_rules_md_path()),
            "dictionary_path": str(_dictionary_md_path()),
            "transactions_db": str(_transactions_db_path()),
            "accounting_db": str(_accounting_db_path()),
        },
    }
