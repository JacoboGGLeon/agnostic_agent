from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional
import os
import unicodedata


def fmt_args(args: Dict[str, Any]) -> str:
    if not args:
        return ""
    parts: List[str] = []
    for key, value in args.items():
        if isinstance(value, str):
            parts.append(f"{key}='{value}'")
        else:
            parts.append(f"{key}={repr(value)}")
    return ", ".join(parts)


def fmt_output(tool_name: str, value: Any, json_default: Optional[Callable[[Any], Any]] = None) -> str:
    if isinstance(value, bool):
        return "Si" if value else "No"

    if isinstance(value, (dict, list, tuple, set)):
        try:
            kwargs = {"ensure_ascii": False, "indent": 2}
            if json_default is not None:
                kwargs["default"] = json_default
            return json.dumps(value, **kwargs)
        except Exception:
            return str(value)

    return str(value)


def summarize_tool_runs(
    user_text: str,
    runs: List[Dict[str, Any]],
    *,
    json_default: Optional[Callable[[Any], Any]] = None,
) -> str:
    if not runs:
        return (
            "No se invoco ninguna herramienta. "
            "No puedo responder con garantias a la pregunta solo con razonamiento interno."
        )

    parts = ["Summary based on tools (no hallucinations)"]

    for idx, run in enumerate(runs, start=1):
        tool_name = run["name"]
        args = run["args"]
        output = run["output"]
        arg_str = fmt_args(args)

        if tool_name == "search_knowledge_base" and isinstance(output, list):
            parts.append(f"\n### Search results (`{arg_str}`)")
            if not output:
                parts.append("_(No relevant results)_")
            else:
                js_noise = [
                    x for x in output if isinstance(x, str) and "[object Object]" in x
                ]
                if js_noise and len(js_noise) == len(output):
                    parts.append(
                        "_(search_knowledge_base devolvio objetos no serializados "
                        "(`[object Object]`). Revisa el contrato de salida de tools.)_"
                    )
                    continue
                if output and not any(isinstance(x, dict) for x in output):
                    try:
                        kwargs = {"ensure_ascii": False}
                        if json_default is not None:
                            kwargs["default"] = json_default
                        raw_dump = json.dumps(output, **kwargs)
                    except Exception:
                        raw_dump = str(output)
                    parts.append(f"```json\n{raw_dump}\n```")
                    continue

                for idx, item in enumerate(output, start=1):
                    if not isinstance(item, dict):
                        parts.append(f"- {idx}. {item}")
                        continue
                    source = item.get("source", "unknown")
                    score = item.get("score")
                    excerpt = item.get("excerpt", "")
                    if score is None:
                        parts.append(f"- **{idx}. source={source}**")
                    else:
                        parts.append(f"- **{idx}. source={source}, score={score:.4f}**")
                    if excerpt:
                        parts.append(f"  - excerpt: {excerpt}")
            continue

        rendered_output = fmt_output(tool_name, output, json_default=json_default)
        parts.append(f"- `{tool_name}({arg_str})`:")
        if isinstance(output, (dict, list)):
            parts.append(f"```json\n{rendered_output}\n```")
        else:
            parts.append(f"`{rendered_output}`")

    return "\n".join(parts)


def _pick_entity_id(args: Dict[str, Any], output: Any) -> str:
    preferred_keys = [
        "id",
        "entity_id",
        "record_id",
        "item_id",
        "document_id",
        "ticket_id",
        "task_id",
        "codigo",
        "key",
        "name",
    ]
    for key in preferred_keys:
        value = args.get(key)
        if value is None and isinstance(output, dict):
            value = output.get(key)
        if value not in (None, ""):
            return f"{key}={value}"
    for key, value in args.items():
        if key.endswith("_id") and value not in (None, ""):
            return f"{key}={value}"
    if isinstance(output, dict):
        for key, value in output.items():
            if key.endswith("_id") and value not in (None, ""):
                return f"{key}={value}"
    return ""


def summarize_tool_runs_compact(runs: List[Dict[str, Any]]) -> str:
    if not runs:
        return "No se ejecutaron herramientas."

    lines: List[str] = [f"Se ejecutaron {len(runs)} tools."]
    for idx, run in enumerate(runs, start=1):
        name = str(run.get("name", "tool"))
        args = run.get("args", {}) or {}
        output = run.get("output")

        entity = _pick_entity_id(args if isinstance(args, dict) else {}, output)

        if isinstance(output, dict):
            if "error" in output:
                status = f"error={output.get('error')}"
            elif "status" in output:
                status = f"status={output.get('status')}"
            elif "ok" in output:
                status = "ok=true" if bool(output.get("ok")) else "ok=false"
            else:
                status = "resultado=dict"
        elif output is None:
            status = "resultado=vacio"
        else:
            status = f"resultado={type(output).__name__}"

        suffix = f" {entity}" if entity else ""
        lines.append(f"{idx}. {name}{suffix} -> {status}")

    return "\n".join(lines)


def looks_like_technical_answer(text: str) -> bool:
    if not isinstance(text, str):
        return False
    low = text.strip().lower()
    if not low:
        return False
    technical_markers = [
        "se ejecutaron",
        "tool_call_id",
        "output_type",
        "step 1",
        "resultado=dict",
        "args:",
    ]
    hits = sum(1 for marker in technical_markers if marker in low)
    if hits >= 2:
        return True
    lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
    numbered_tool_lines = sum(
        1 for ln in lines if re.match(r"^\d+\.\s+\w+", ln) and "->" in ln
    )
    return numbered_tool_lines >= 3


def _pick_status(output: Any) -> str:
    if isinstance(output, dict):
        if "error" in output:
            return f"error={output.get('error')}"
        for key in ("status", "result", "message"):
            if key in output and output.get(key) not in (None, ""):
                return f"{key}={output.get(key)}"
        if "ok" in output:
            return "ok=true" if bool(output.get("ok")) else "ok=false"
        return "resultado=estructurado"
    if output is None:
        return "resultado=vacio"
    return f"resultado={type(output).__name__}"


def _extract_subqueries_from_prompt(user_prompt: str) -> List[Dict[str, str]]:
    text = (user_prompt or "").strip()
    if not text:
        return []

    rows: List[Dict[str, str]] = []

    # Case 1: prompt contains multiple JSON objects (common in batch structured inputs).
    json_chunks = re.findall(r"\{[^{}]+\}", text)
    if len(json_chunks) >= 2:
        for idx, chunk in enumerate(json_chunks, start=1):
            label = f"Subconsulta {idx}"
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    cid = obj.get("credito_id")
                    if cid:
                        label = f"Credito {cid}"
            except Exception:
                pass
            rows.append({"label": label, "text": chunk})
        return rows

    # Case 2: multi-question prompt separated by '?'.
    questions = [q.strip() + "?" for q in text.split("?") if q.strip()]
    if len(questions) >= 2:
        for idx, q in enumerate(questions, start=1):
            rows.append({"label": f"Pregunta {idx}", "text": q})
        return rows

    # Case 3: fallback single request.
    return [{"label": "Solicitud", "text": text}]


def _extract_entity_from_text(text: str, known_entities: List[str]) -> str:
    """
    Map subquery text to an already-observed entity key (e.g., 'credito_id=LOC-0010')
    without hardcoding domain-specific patterns.
    """
    t = (text or "").strip().lower()
    if not t or not known_entities:
        return ""

    for entity in known_entities:
        if "=" not in entity:
            continue
        _, value = entity.split("=", 1)
        if str(value).strip().lower() and str(value).strip().lower() in t:
            return entity
    return ""


def _fmt_number(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _truncate(text: str, max_len: int = 220) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[:max_len].rstrip() + "..."


def _fmt_row_preview(row: Any, max_items: int = 4) -> str:
    if isinstance(row, dict):
        parts: List[str] = []
        for idx, (k, v) in enumerate(row.items()):
            if idx >= max_items:
                break
            parts.append(f"{k}={v}")
        return ", ".join(parts)
    if isinstance(row, (list, tuple)):
        return ", ".join(str(x) for x in row[:max_items])
    return str(row)


def _normalize_query_text(text: str) -> str:
    t = unicodedata.normalize("NFKD", str(text or ""))
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _rows_to_dicts(columns: List[Any], rows: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    col_names = [str(c) for c in columns]
    for row in rows or []:
        if isinstance(row, dict):
            out.append({str(k): v for k, v in row.items()})
            continue
        if isinstance(row, (list, tuple)):
            out.append({col_names[idx]: row[idx] for idx in range(min(len(col_names), len(row)))})
    return out


def _looks_like_top_n_request(query_text: str) -> bool:
    q = _normalize_query_text(query_text)
    return bool(re.search(r"\btop\s+\d+\b", q)) or any(tok in q for tok in ["mas alto", "más alto", "mas grande", "mayor", "ranking"])


def _looks_like_aggregate_request(query_text: str) -> bool:
    q = _normalize_query_text(query_text)
    return any(tok in q for tok in ["promedio", "avg", "suma", "sum", "count", "cuantos", "cuántos", "agrupa", "group by", "por tipo", "por estatus"])


def _render_result_table(rows: List[Dict[str, Any]], columns: List[Any], max_rows: int = 5) -> str:
    if not rows:
        return ""
    lines: List[str] = []
    for idx, row in enumerate(rows[:max_rows], start=1):
        parts = [f"{key}={value}" for key, value in row.items()]
        lines.append(f"{idx}. " + " | ".join(parts))
    return "\n".join(lines)


def _infer_sql_response_shape(query_text: str, output: Dict[str, Any]) -> Dict[str, Any]:
    execution = output.get("execution") if isinstance(output.get("execution"), dict) else {}
    columns = execution.get("columns") if isinstance(execution.get("columns"), list) else output.get("columns")
    rows = execution.get("rows") if isinstance(execution.get("rows"), list) else output.get("rows")
    row_count = execution.get("row_count") if isinstance(execution.get("row_count"), int) else (len(rows) if isinstance(rows, list) else 0)
    columns = columns if isinstance(columns, list) else []
    rows = rows if isinstance(rows, list) else []
    row_dicts = _rows_to_dicts(columns, rows)
    q = _normalize_query_text(query_text)

    if row_count == 1 and row_dicts:
        return {"shape": "single_record", "rows": row_dicts, "columns": columns, "row_count": row_count}
    if _looks_like_top_n_request(q) and row_dicts:
        return {"shape": "top_n_list", "rows": row_dicts, "columns": columns, "row_count": row_count}
    if _looks_like_aggregate_request(q) and row_dicts:
        if row_count == 1 and len(columns) == 1:
            return {"shape": "aggregate_scalar", "rows": row_dicts, "columns": columns, "row_count": row_count}
        return {"shape": "grouped_aggregate", "rows": row_dicts, "columns": columns, "row_count": row_count}
    if row_dicts:
        return {"shape": "record_list", "rows": row_dicts, "columns": columns, "row_count": row_count}
    return {"shape": "unknown", "rows": [], "columns": columns, "row_count": row_count}


def _render_sql_shape(shape_info: Dict[str, Any], output: Dict[str, Any]) -> str:
    shape = str(shape_info.get("shape") or "unknown")
    rows = shape_info.get("rows") if isinstance(shape_info.get("rows"), list) else []
    row_count = int(shape_info.get("row_count") or 0)
    chosen_table = output.get("chosen_table")

    if shape == "single_record" and rows:
        row = rows[0]
        parts = [f"{key}={value}" for key, value in row.items()]
        return "Encontré 1 registro" + (f" en {chosen_table}" if chosen_table else "") + ": " + ", ".join(parts) + "."

    if shape == "top_n_list" and rows:
        title = f"Top {min(row_count, len(rows))} resultado(s)" + (f" en {chosen_table}" if chosen_table else "")
        return title + ":\n" + _render_result_table(rows, shape_info.get("columns") or [], max_rows=min(10, len(rows)))

    if shape == "aggregate_scalar" and rows:
        row = rows[0]
        parts = [f"{key}={value}" for key, value in row.items()]
        return "Resultado agregado: " + ", ".join(parts) + "."

    if shape == "grouped_aggregate" and rows:
        return "Resultados agregados:\n" + _render_result_table(rows, shape_info.get("columns") or [], max_rows=min(10, len(rows)))

    if shape == "record_list" and rows:
        preview_rows = rows[: min(5, len(rows))]
        return f"Encontré {row_count} registro(s)" + (f" en {chosen_table}" if chosen_table else "") + ":\n" + _render_result_table(preview_rows, shape_info.get("columns") or [], max_rows=len(preview_rows))

    return ""


def _score_run_for_user_answer(run: Dict[str, Any]) -> int:
    """
    Prefer runs that contain richer structured evidence over generic status-only runs.
    """
    output = run.get("output")
    if not isinstance(output, dict):
        return 1

    score = 1
    if "error" in output:
        score -= 2
    if isinstance(output.get("saldo"), dict):
        score += 4
    if isinstance(output.get("saneamiento"), dict):
        score += 4
    if "tasa_saneamiento" in output:
        score += 2
    if isinstance(output.get("execution"), dict):
        score += 2
    if isinstance(output.get("rows"), list):
        score += 2
    if isinstance(output.get("status"), str):
        score += 1
    return score


def _generic_output_summary(output: Any) -> str:
    if output is None:
        return "No hubo salida util para responder."
    if isinstance(output, str):
        t = _truncate(output, 240)
        return t if t else "No hubo salida util para responder."
    if isinstance(output, list):
        if not output:
            return "La herramienta no devolvió resultados."
        first = output[0]
        preview = _fmt_row_preview(first)
        return f"La herramienta devolvió {len(output)} resultado(s). Ejemplo: {preview}."
    if isinstance(output, dict):
        if output.get("error"):
            return f"La herramienta reportó error: {output.get('error')}."
        # Prefer human-meaningful keys before raw status flags.
        for key in ("answer", "summary", "result", "message", "status", "descripcion", "detalle"):
            val = output.get(key)
            if val not in (None, ""):
                return f"{key}: {_truncate(str(val), 240)}"
        # If dict carries rows-like payloads, expose them.
        if isinstance(output.get("rows"), list):
            rows = output.get("rows")
            if rows:
                return f"Se obtuvieron {len(rows)} fila(s). Ejemplo: {_fmt_row_preview(rows[0])}."
            return "La consulta se ejecutó, pero no devolvió filas."
        # Generic compact dict preview.
        preview_parts: List[str] = []
        for idx, (k, v) in enumerate(output.items()):
            if idx >= 4:
                break
            if isinstance(v, (dict, list)):
                preview_parts.append(f"{k}=estructurado")
            else:
                preview_parts.append(f"{k}={v}")
        if preview_parts:
            return "Datos clave: " + ", ".join(preview_parts) + "."
    return f"Resultado disponible: {type(output).__name__}."


def _parse_output_json_dict(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _norm_status_key(text: str) -> str:
    n = unicodedata.normalize("NFKD", str(text or ""))
    n = "".join(ch for ch in n if not unicodedata.combining(ch)).lower().strip()
    n = re.sub(r"\s+", " ", n)
    return n


def _extract_credit_id_from_text(text: str) -> str:
    t = str(text or "")
    m = re.search(r"credito_id\s*=\s*['\"]([^'\"]+)['\"]", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().upper()
    m = re.search(r"\bLOC-\d{3,}\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(0).strip().upper()
    return ""


def _extract_credit_ids_from_subqueries(subqueries: List[str]) -> List[str]:
    ids: List[str] = []
    for sq in subqueries or []:
        txt = str(sq or "").strip()
        if not txt:
            continue
        found = ""
        if txt.startswith("{") and txt.endswith("}"):
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    found = str(obj.get("credito_id") or "").strip().upper()
            except Exception:
                found = ""
        if not found:
            found = _extract_credit_id_from_text(txt)
        if found and found not in ids:
            ids.append(found)
    return ids


def _finance_reconciliation_answer(
    user_prompt: str,
    runs: List[Dict[str, Any]],
    analyzer_subqueries: Optional[List[str]] = None,
) -> str:
    requested_order: List[str] = []
    for cid in _extract_credit_ids_from_subqueries(analyzer_subqueries or []):
        if cid not in requested_order:
            requested_order.append(cid)
    prompt_ids = [m.upper() for m in re.findall(r"\bLOC-\d{3,}\b", user_prompt or "", flags=re.IGNORECASE)]
    for cid in prompt_ids:
        if cid not in requested_order:
            requested_order.append(cid)

    by_credit: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for run in runs:
        name = str(run.get("name", ""))
        args = run.get("args", {}) if isinstance(run.get("args"), dict) else {}
        q = str(args.get("query", "") or "")
        cid = _extract_credit_id_from_text(q)
        if not cid:
            output_dict = _parse_output_json_dict(run.get("output"))
            normalized_query = str(output_dict.get("normalized_query", "") or "")
            cid = _extract_credit_id_from_text(normalized_query)
        if not cid:
            continue
        by_credit.setdefault(cid, {})
        if name == "query_transactions_db":
            by_credit[cid]["tx"] = run
        elif name == "query_accounting_db":
            by_credit[cid]["acc"] = run
    if not requested_order:
        requested_order = list(by_credit.keys())
    if not requested_order:
        return ""

    def _idx(cols: List[Any], target: str) -> int:
        for i, c in enumerate(cols):
            if str(c).lower() == target:
                return i
        return -1

    rates = {
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

    blocks: List[str] = []
    for credito_id in requested_order:
        pair = by_credit.get(credito_id, {})
        tx_run = pair.get("tx")
        acc_run = pair.get("acc")
        if not tx_run or not acc_run:
            continue

        tx_out = _parse_output_json_dict(tx_run.get("output"))
        acc_out = _parse_output_json_dict(acc_run.get("output"))
        if not tx_out or not acc_out:
            continue

        tx_cols = tx_out.get("columns") if isinstance(tx_out.get("columns"), list) else []
        tx_rows = tx_out.get("rows") if isinstance(tx_out.get("rows"), list) else []
        acc_cols = acc_out.get("columns") if isinstance(acc_out.get("columns"), list) else []
        acc_rows = acc_out.get("rows") if isinstance(acc_out.get("rows"), list) else []
        if not tx_cols or not acc_cols or not acc_rows:
            continue

        tipo_i = _idx(tx_cols, "tipo")
        monto_i = _idx(tx_cols, "monto")
        saldo_i = _idx(acc_cols, "saldo_total")
        estatus_i = _idx(acc_cols, "estatus")
        saneamiento_i = _idx(acc_cols, "saneamiento_calculado")
        if min(tipo_i, monto_i, saldo_i, estatus_i, saneamiento_i) < 0:
            continue

        totals = {"DESEMBOLSO": 0.0, "PAGO": 0.0, "PENALIZACION": 0.0, "DESCUENTO": 0.0}
        for row in tx_rows:
            if not isinstance(row, (list, tuple)) or len(row) <= max(tipo_i, monto_i):
                continue
            tipo = str(row[tipo_i]).strip().upper()
            try:
                monto = float(row[monto_i])
            except Exception:
                monto = 0.0
            if tipo in totals:
                totals[tipo] += monto

        acc = acc_rows[0]
        if not isinstance(acc, (list, tuple)) or len(acc) <= max(saldo_i, estatus_i, saneamiento_i):
            continue
        saldo_reportado = float(acc[saldo_i])
        estatus = str(acc[estatus_i])
        saneamiento_reportado = float(acc[saneamiento_i])

        saldo_esperado = totals["DESEMBOLSO"] - totals["PAGO"] + totals["PENALIZACION"] - totals["DESCUENTO"]
        diff_saldo = saldo_esperado - saldo_reportado
        tasa = rates.get(_norm_status_key(estatus), 0.0)
        reserva_esperada = saldo_reportado * tasa
        diff_reserva = reserva_esperada - saneamiento_reportado

        ok_saldo = abs(diff_saldo) < 0.01
        ok_reserva = abs(diff_reserva) < 0.01
        estado = "CUADRADO (100% Match)" if ok_saldo and ok_reserva else "DRIFT DETECTADO"

        blocks.append(
            (
                f"Conciliacion del credito {credito_id}: {estado}.\n"
                f"- Saldo esperado: {saldo_esperado:.2f} | Saldo reportado: {saldo_reportado:.2f} | Diferencia: {diff_saldo:.2f}\n"
                f"- Reserva esperada: {reserva_esperada:.2f} (tasa {tasa*100:.2f}% por estatus '{estatus}') | "
                f"Reserva reportada: {saneamiento_reportado:.2f} | Diferencia: {diff_reserva:.2f}"
            )
        )

    return "\n\n".join(blocks) if blocks else ""


def _summarize_single_run_natural(run: Dict[str, Any]) -> str:
    output = run.get("output")
    parsed = _parse_output_json_dict(output)
    if parsed:
        output = parsed

    if isinstance(output, dict) and (
        isinstance(output.get("saldo"), dict) or isinstance(output.get("saneamiento"), dict)
    ):
        status = output.get("status") or ("OK" if output.get("ok") else "Revisar")
        saldo = output.get("saldo") if isinstance(output.get("saldo"), dict) else {}
        saneamiento = output.get("saneamiento") if isinstance(output.get("saneamiento"), dict) else {}
        d_saldo = _fmt_number(saldo.get("diferencia", 0))
        d_san = _fmt_number(saneamiento.get("diferencia", 0))
        return f"{status}. Diferencia de saldo: {d_saldo}. Diferencia de saneamiento: {d_san}."

    if isinstance(output, dict) and "tasa_saneamiento" in output:
        tasa = output.get("tasa_saneamiento")
        if tasa is not None:
            try:
                return f"Tasa de saneamiento esperada: {float(tasa) * 100:.2f}%."
            except Exception:
                return f"Tasa de saneamiento esperada: {tasa}."
        return "Se obtuvo la tasa de saneamiento esperada."

    if isinstance(output, dict) and (
        isinstance(output.get("execution"), dict) or isinstance(output.get("rows"), list)
    ):
        if not output.get("ok"):
            err = output.get("error") or "fallo en consulta SQL"
            return f"No pude consultar la base de datos: {err}."

        req_text = str(output.get("user_request") or "")
        where_clauses = output.get("where_clauses") or []
        loc_match = re.search(r"\bLOC-\d{3,}\b", req_text, flags=re.IGNORECASE)
        loc_id = loc_match.group(0).upper() if loc_match else ""
        if not loc_id and output.get("entity_id"):
            loc_id = str(output.get("entity_id")).strip().upper()

        chosen_table = output.get("chosen_table")
        db_label = os.path.basename(str(output.get("db_path") or ""))
        execution = output.get("execution") if isinstance(output.get("execution"), dict) else {}
        generated_sql = str(output.get("generated_sql") or "")
        sql_supposed = str(output.get("sql_supposed") or "")
        filter_signals = " ".join([generated_sql, sql_supposed] + [str(x) for x in where_clauses])
        has_entity_filter = bool(loc_id and loc_id.lower() in filter_signals.lower())
        suspicious_table = bool(
            chosen_table and str(chosen_table).lower() not in {"estados_cuenta", "movimientos"}
        )
        suspicious_db = "embeddings.db" in db_label.lower()
        if loc_id and not has_entity_filter and (suspicious_table or suspicious_db):
            return (
                f"No pude filtrar por {loc_id} en la base `{db_label}`. "
                "Necesito consultar `session/contabilidad.db` o `session/transacciones.db` para ese credito."
            )
        if execution and execution.get("ok"):
            rows = execution.get("rows") if isinstance(execution.get("rows"), list) else []
            row_count = execution.get("row_count")
            if isinstance(row_count, int):
                if loc_id and row_count == 0:
                    return f"No encontre registros para {loc_id}."
                shape_info = _infer_sql_response_shape(req_text, output)
                rendered = _render_sql_shape(shape_info, output)
                if rendered:
                    return rendered
                table_txt = f" en {chosen_table}" if chosen_table else ""
                return f"La consulta devolvió {row_count} registro(s){table_txt}."
        if loc_id and not has_entity_filter:
            return (
                f"No pude filtrar por {loc_id} en la base `{db_label}`. "
                "Necesito consultar `session/contabilidad.db` o `session/transacciones.db` para ese credito."
            )
        if chosen_table:
            return f"Preparé la consulta sobre {chosen_table}, pero no hay resultados ejecutados para responder con datos."
        return "Pude generar la consulta SQL, pero no hay resultados ejecutados para responder con datos."

    if isinstance(output, list):
        if isinstance(output, list) and output:
            snippets: List[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                excerpt = item.get("excerpt") or item.get("content") or item.get("text") or ""
                txt = _truncate(str(excerpt), 200)
                if txt:
                    snippets.append(txt)
                if len(snippets) >= 2:
                    break
            if snippets:
                if len(snippets) == 1:
                    return f"Según la evidencia encontrada: {snippets[0]}"
                return f"Según la evidencia encontrada: {snippets[0]} Además: {snippets[1]}"
            return "Encontre resultados relevantes en la base de conocimiento, pero sin extractos legibles."
        return "No encontre resultados relevantes en la base de conocimiento."

    status = _pick_status(output)
    generic = _generic_output_summary(output)
    if generic:
        return generic
    return f"Resultado: {status}."


def _build_response_items(
    user_prompt: str,
    runs: List[Dict[str, Any]],
    analyzer_subqueries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    finance_answer = _finance_reconciliation_answer(user_prompt, runs, analyzer_subqueries)
    if finance_answer:
        return {
            "kind": "finance_reconciliation",
            "user_prompt": user_prompt,
            "items": [
                {
                    "label": "Conciliacion",
                    "message": block.strip(),
                    "entity": "",
                    "source": "finance",
                }
                for block in finance_answer.split("\n\n")
                if block.strip()
            ],
            "errors": 0,
            "findings": [],
        }

    errors = 0
    findings: List[str] = []
    by_entity: Dict[str, List[Dict[str, Any]]] = {}
    no_entity_runs: List[Dict[str, Any]] = []

    for run in runs:
        name = str(run.get("name", "tool"))
        args = run.get("args", {}) or {}
        output = run.get("output")
        entity = _pick_entity_id(args if isinstance(args, dict) else {}, output)
        status = _pick_status(output)
        if status.startswith("error="):
            errors += 1
        detail = f" ({entity})" if entity else ""
        findings.append(f"{name}{detail}: {status}")
        if entity:
            by_entity.setdefault(entity, []).append(run)
        else:
            no_entity_runs.append(run)

    subqueries = _extract_subqueries_from_prompt(user_prompt)
    known_entities = list(by_entity.keys())
    items: List[Dict[str, Any]] = []
    for idx, sq in enumerate(subqueries, start=1):
        sq_text = sq.get("text", "")
        sq_label = sq.get("label", f"Subconsulta {idx}")
        entity = _extract_entity_from_text(sq_text, known_entities)
        try:
            parsed = json.loads(sq_text) if sq_text.startswith("{") else {}
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if str(k).endswith("_id") and v not in (None, ""):
                        candidate = f"{k}={v}"
                        if candidate in by_entity:
                            entity = candidate
                            break
        except Exception:
            pass

        message = ""
        if entity and entity in by_entity:
            entity_runs = by_entity.get(entity, [])
            ranked = sorted(entity_runs, key=_score_run_for_user_answer, reverse=True)
            message = _summarize_single_run_natural(ranked[0])
        elif no_entity_runs:
            # Map free-form queries to no-entity runs in order.
            run_idx = idx - 1
            if run_idx < len(no_entity_runs):
                message = _summarize_single_run_natural(no_entity_runs[run_idx])

        if not message:
            message = "No hubo evidencia suficiente de tools para esta subconsulta."

        # If output is rate-only and query requested more detail, make gap explicit in neutral terms.
        low_sq = sq_text.lower()
        asked_diffs = ("diferencia" in low_sq) or ("saldo" in low_sq and "saneamiento" in low_sq)
        if asked_diffs and "Tasa de saneamiento esperada" in message:
            message = (
                f"{message} Aun no tengo evidencia suficiente para diferencias completas de saldo y saneamiento "
                "en esta subconsulta."
            )

        label_out = entity if entity else sq_label
        items.append(
            {
                "label": label_out,
                "message": message,
                "entity": entity,
                "source": "subquery",
            }
        )

    if not items and no_entity_runs:
        for idx, run in enumerate(no_entity_runs, start=1):
            items.append(
                {
                    "label": f"Resultado {idx}",
                    "message": _summarize_single_run_natural(run),
                    "entity": "",
                    "source": "fallback",
                }
            )

    return {
        "kind": "tool_evidence",
        "user_prompt": user_prompt,
        "items": items,
        "errors": errors,
        "findings": findings,
        "subqueries": subqueries,
    }


def build_response_bundle(
    user_prompt: str,
    runs: List[Dict[str, Any]],
    analyzer_subqueries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not runs:
        return {
            "kind": "empty",
            "user_prompt": user_prompt,
            "items": [
                {
                    "label": "Solicitud",
                    "message": "No se obtuvo evidencia de herramientas para resolver la solicitud.",
                    "entity": "",
                    "source": "empty",
                }
            ],
            "errors": 0,
            "findings": [],
        }
    return _build_response_items(user_prompt, runs, analyzer_subqueries)


def render_response_bundle(bundle: Dict[str, Any], level: str = "user") -> str:
    items = bundle.get("items") if isinstance(bundle.get("items"), list) else []
    errors = int(bundle.get("errors") or 0)
    findings = bundle.get("findings") if isinstance(bundle.get("findings"), list) else []
    kind = str(bundle.get("kind") or "tool_evidence")

    if level == "user":
        lines: List[str] = []
        for idx, item in enumerate(items, start=1):
            message = str(item.get("message") or "").strip()
            label = str(item.get("label") or "").strip()
            if not message:
                continue
            if kind == "finance_reconciliation":
                lines.append(message)
                continue
            if len(items) == 1 and label.lower() in {"solicitud", "resultado 1"}:
                lines.append(message)
            else:
                lines.append(f"{idx}. {label}: {message}")
        if errors > 0:
            lines.append(f"Detecte {errors} ejecuciones con error en la evidencia disponible.")
        return "\n\n".join(line for line in lines if line).strip()

    if level == "dev":
        lines = ["Respuesta derivada de evidencia verificada."]
        if items:
            lines.append("")
            lines.append("Resultado sintetizado:")
            for idx, item in enumerate(items, start=1):
                label = str(item.get("label") or f"Resultado {idx}").strip()
                message = str(item.get("message") or "").strip()
                if not message:
                    continue
                lines.append(f"{idx}. {label}: {message}")
        if findings:
            lines.append("")
            lines.append("Hallazgos operativos:")
            for idx, finding in enumerate(findings, start=1):
                lines.append(f"{idx}. {finding}")
        if errors == 0:
            lines.append("")
            lines.append("Estado de ejecucion: sin errores detectados.")
        else:
            lines.append("")
            lines.append(f"Estado de ejecucion: {errors} error(es) detectado(s).")
        return "\n".join(lines).strip()

    lines = ["Representacion profunda basada en evidencia verificada."]
    lines.append("")
    lines.append(f"Tipo de respuesta: {kind}")
    lines.append(f"Items sintetizados: {len(items)}")
    lines.append(f"Errores detectados: {errors}")
    if items:
        lines.append("")
        lines.append("Detalle sintetizado:")
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"{idx}. label={item.get('label')} | entity={item.get('entity') or '-'} | "
                f"source={item.get('source') or '-'} | message={item.get('message')}"
            )
    if findings:
        lines.append("")
        lines.append("Hallazgos base:")
        for idx, finding in enumerate(findings, start=1):
            lines.append(f"{idx}. {finding}")
    return "\n".join(lines).strip()


def build_agnostic_user_answer(
    user_prompt: str,
    runs: List[Dict[str, Any]],
    analyzer_subqueries: Optional[List[str]] = None,
) -> str:
    bundle = build_response_bundle(user_prompt, runs, analyzer_subqueries)
    return render_response_bundle(bundle, level="user")
