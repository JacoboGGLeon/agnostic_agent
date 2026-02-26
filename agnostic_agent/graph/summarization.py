from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional


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


def _summarize_single_run_natural(run: Dict[str, Any]) -> str:
    output = run.get("output")

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

        chosen_table = output.get("chosen_table")
        execution = output.get("execution") if isinstance(output.get("execution"), dict) else {}
        if execution and execution.get("ok"):
            rows = execution.get("rows") if isinstance(execution.get("rows"), list) else []
            row_count = execution.get("row_count")
            if isinstance(row_count, int):
                if rows:
                    preview = _fmt_row_preview(rows[0])
                    table_txt = f" de {chosen_table}" if chosen_table else ""
                    return f"Encontré {row_count} registro(s){table_txt}. Ejemplo: {preview}."
                table_txt = f" en {chosen_table}" if chosen_table else ""
                return f"La consulta devolvió {row_count} registro(s){table_txt}."
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


def build_agnostic_user_answer(user_prompt: str, runs: List[Dict[str, Any]]) -> str:
    if not runs:
        return "No se obtuvo evidencia de herramientas para resolver la solicitud."

    total = len(runs)
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

    lines: List[str] = []
    lines.append("Ya lo revise.")
    subqueries = _extract_subqueries_from_prompt(user_prompt)

    if len(subqueries) > 1:
        lines.append("Te respondo punto por punto:")
    else:
        lines.append("Respuesta:")

    known_entities = list(by_entity.keys())
    # Try to answer subquery by subquery using entity-linked runs first.
    emitted = 0
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
        lines.append(f"{idx}. {label_out}: {message}")
        emitted += 1

    # If we still have unmatched no-entity runs, append as additional evidence.
    if emitted == 0 and no_entity_runs:
        for idx, run in enumerate(no_entity_runs, start=1):
            lines.append(f"{idx}. {_summarize_single_run_natural(run)}")

    lines.append("")
    if errors == 0:
        lines.append("No detecte errores de ejecucion en la evidencia disponible.")
    else:
        lines.append(f"Detecte {errors} ejecuciones con error; conviene revisar el detalle tecnico en Deep/Dev.")
    lines.append("Si quieres el detalle tecnico completo, lo tienes en Deep/Dev.")

    return "\n".join(lines).strip()
