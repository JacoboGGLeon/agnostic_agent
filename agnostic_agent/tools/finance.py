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


def _normalize_finance_sql(query: str) -> str:
    """
    Normaliza aliases comunes de tablas para tolerar prompts/plans con nombres genéricos.
    """
    normalized = query or ""
    normalized = re.sub(r"\bcreditos\b", "estados_cuenta", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\btransacciones\b", "movimientos", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bcodigo\b", "credito_id", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bsaldo\b", "saldo_total", normalized, flags=re.IGNORECASE)
    return normalized


def _run_query(db_path: Path, query: str) -> str:
    normalized_query = _normalize_finance_sql(query)
    if not _is_read_only_sql(normalized_query):
        return "Error SQL: solo se permiten consultas SELECT de solo lectura."
    if not db_path.exists():
        return f"Error SQL: no se encontro la base de datos: {db_path}"
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(normalized_query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in (cur.description or [])]
        conn.close()
        return json.dumps(
            {
                "normalized_query": normalized_query,
                "columns": columns,
                "rows": rows,
            },
            ensure_ascii=False,
        )
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
_DICTIONARY_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "entries": None}


def _strict_rules_mode() -> bool:
    return os.getenv("AGNOSTIC_FIN_STRICT_RULES", "0").lower() in ("1", "true", "yes", "on")


def _parse_percent_to_rate(raw: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", raw or "")
    if not match:
        return None
    return float(match.group(1)) / 100.0


def get_known_finance_statuses() -> List[str]:
    rates = _get_runtime_rates()
    known = sorted(rates.keys())
    return [status for status in known]


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


def _load_rule_rows_from_rules_md(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8", errors="ignore")
    rows: List[Dict[str, Any]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "Estatus" in stripped:
            continue
        if re.fullmatch(r"\|[-\s|:]+\|?", stripped):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 3:
            continue
        rows.append(
            {
                "estatus": parts[0],
                "dias_mora": parts[1],
                "tasa_raw": parts[2],
                "tasa": _parse_percent_to_rate(parts[2]),
                "source_line": stripped,
            }
        )
    return rows


def _load_dictionary_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8", errors="ignore")
    entries: List[Dict[str, Any]] = []
    current_source = ""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current_source = stripped.lstrip("#").strip()
            continue
        if not stripped.startswith("|") or "Columna" in stripped:
            continue
        if re.fullmatch(r"\|[-\s|:]+\|?", stripped):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 3:
            continue
        term = parts[0].strip("`")
        entries.append(
            {
                "term": term,
                "type": parts[1],
                "definition": parts[2],
                "section": current_source,
                "source_line": stripped,
            }
        )
    return entries


def _get_dictionary_entries() -> List[Dict[str, Any]]:
    dict_path = _dictionary_md_path()
    mtime = dict_path.stat().st_mtime if dict_path.exists() else None
    if (
        _DICTIONARY_CACHE.get("path") == str(dict_path)
        and _DICTIONARY_CACHE.get("mtime") == mtime
        and isinstance(_DICTIONARY_CACHE.get("entries"), list)
    ):
        return _DICTIONARY_CACHE["entries"]
    entries = _load_dictionary_entries(dict_path)
    _DICTIONARY_CACHE["path"] = str(dict_path)
    _DICTIONARY_CACHE["mtime"] = mtime
    _DICTIONARY_CACHE["entries"] = entries
    return entries


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


@tool(mode="public")
def lookup_finance_rule(query: str, estatus: str = "") -> Dict[str, Any]:
    """
    Busca evidencia semántica en rules.md para reglas financieras o de saneamiento.
    """
    rules_path = _rules_md_path()
    rows = _load_rule_rows_from_rules_md(rules_path)
    query_norm = _normalize_text(query)
    estatus_norm = _normalize_text(estatus)
    best: Dict[str, Any] = {}
    best_score = -1
    for row in rows:
        status_norm = _normalize_text(row.get("estatus", ""))
        score = 0
        if estatus_norm and status_norm == estatus_norm:
            score += 4
        if status_norm and status_norm in query_norm:
            score += 3
        if any(token and token in query_norm for token in [_normalize_text(row.get("dias_mora", "")), _normalize_text(row.get("tasa_raw", ""))]):
            score += 1
        if score > best_score:
            best = row
            best_score = score
    found = bool(best) and best_score > 0
    snippets = [best.get("source_line", "")] if found else []
    return {
        "found": found,
        "query": query,
        "estatus": estatus or best.get("estatus", ""),
        "matched_rule": best if found else {},
        "source_path": str(rules_path),
        "evidence_snippets": snippets,
        "confidence": round(best_score / 5.0, 2) if found else 0.0,
    }


@tool(mode="public")
def lookup_finance_dictionary(term: str) -> Dict[str, Any]:
    """
    Busca definiciones semánticas en dictionary.md para términos del mundo financiero.
    """
    dict_path = _dictionary_md_path()
    entries = _get_dictionary_entries()
    term_norm = _normalize_text(term)
    best: Dict[str, Any] = {}
    best_score = -1
    for entry in entries:
        entry_term = _normalize_text(entry.get("term", ""))
        entry_definition = _normalize_text(entry.get("definition", ""))
        score = 0
        if term_norm == entry_term:
            score += 4
        if term_norm and term_norm in entry_term:
            score += 3
        if term_norm and term_norm in entry_definition:
            score += 2
        if score > best_score:
            best = entry
            best_score = score
    found = bool(best) and best_score > 0
    return {
        "found": found,
        "term": term,
        "matched_term": best.get("term", "") if found else "",
        "definition": best.get("definition", "") if found else "",
        "section": best.get("section", "") if found else "",
        "source_path": str(dict_path),
        "confidence": round(best_score / 4.0, 2) if found else 0.0,
        "evidence_snippets": [best.get("source_line", "")] if found else [],
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


def _normalize_credito_id(raw_credito_id: str) -> str:
    """
    Normaliza entradas de credito_id para tolerar formatos tabulares del planner/LLM.
    Ejemplos:
    - "LOC-0004" -> "LOC-0004"
    - "0 LOC-0004" -> "LOC-0004"
    - "credito_id=LOC-0004" -> "LOC-0004"
    """
    raw = (raw_credito_id or "").strip()
    if not raw:
        return ""
    match = re.search(r"\b(LOC-\d{4,})\b", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return raw.upper()


@tool(mode="public")
def reconcile_credit_accounting(credito_id: str, balance: str = "") -> Dict[str, Any]:
    """
    Concilia un credito 1-a-1 y valida saldo y saneamiento.
    """
    normalized_credito_id = _normalize_credito_id(credito_id)
    if not normalized_credito_id:
        return {"ok": False, "credito_id": credito_id, "error": "credito_id vacio o invalido."}

    try:
        tx_rows = _fetch_transactions(normalized_credito_id)
        saldo_total, estatus, saneamiento_calculado = _fetch_accounting(normalized_credito_id)
    except Exception as exc:
        return {
            "ok": False,
            "credito_id": normalized_credito_id,
            **({"credito_id_input": credito_id} if normalized_credito_id != credito_id else {}),
            "error": str(exc),
        }

    input_balance_raw = (balance or "").strip()
    input_balance = None
    input_balance_ok = None
    input_balance_diff = None
    input_balance_error = None
    if input_balance_raw:
        try:
            input_balance = float(input_balance_raw.replace(",", ""))
            input_balance_diff = round(saldo_total - input_balance, 2)
            input_balance_ok = abs(input_balance_diff) < 0.01
        except Exception as exc:
            input_balance_error = f"No se pudo parsear balance='{input_balance_raw}': {exc}"
            input_balance_ok = False

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
    all_ok = saldo_ok and saneamiento_ok and (input_balance_ok is not False)
    status = "CUADRADO (100% Match)" if all_ok else "DRIFT DETECTADO"

    return {
        "ok": True,
        "credito_id": normalized_credito_id,
        **({"credito_id_input": credito_id} if normalized_credito_id != credito_id else {}),
        "estatus": estatus,
        "status": status,
        "validaciones": {
            "saldo_ok": saldo_ok,
            "saneamiento_ok": saneamiento_ok,
            **({"input_balance_ok": input_balance_ok} if input_balance_raw else {}),
        },
        "flujos": totals,
        "saldo": {
            "reportado": round(saldo_total, 2),
            "esperado": round(saldo_esperado, 2),
            "diferencia": diff_saldo,
        },
        **(
            {
                "input_balance": {
                    "provided": input_balance_raw,
                    "parsed": input_balance,
                    "diferencia_vs_reportado": input_balance_diff,
                    "ok": input_balance_ok,
                    **({"error": input_balance_error} if input_balance_error else {}),
                }
            }
            if input_balance_raw
            else {}
        ),
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
