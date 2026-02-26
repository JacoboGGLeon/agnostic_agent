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
    """
    User-friendly summary based ONLY on tool outputs.
    """
    if not runs:
        return (
            "No se invoco ninguna herramienta. "
            "No puedo responder con garantias a la pregunta solo con razonamiento interno."
        )

    parts = ["Summary based on tools (no hallucinations)"]

    for run in runs:
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


def summarize_tool_runs_compact(runs: List[Dict[str, Any]]) -> str:
    if not runs:
        return "No se ejecutaron herramientas."

    lines: List[str] = [f"Se ejecutaron {len(runs)} tools."]
    for idx, run in enumerate(runs, start=1):
        name = str(run.get("name", "tool"))
        args = run.get("args", {}) or {}
        output = run.get("output")

        credito_id = ""
        if isinstance(args, dict):
            credito_id = str(args.get("credito_id") or "").strip()
        if not credito_id and isinstance(output, dict):
            credito_id = str(output.get("credito_id") or "").strip()

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

        suffix = f" credito_id={credito_id}" if credito_id else ""
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


def _pick_entity_id(args: Dict[str, Any], output: Any) -> str:
    preferred_keys = [
        "id",
        "entity_id",
        "record_id",
        "item_id",
        "document_id",
        "ticket_id",
        "task_id",
        "credito_id",
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


def build_agnostic_user_answer(user_prompt: str, runs: List[Dict[str, Any]]) -> str:
    if not runs:
        return "No se obtuvo evidencia de herramientas para resolver la solicitud."

    total = len(runs)
    errors = 0
    lines: List[str] = []
    lines.append("## Resultado")
    lines.append(f"Se procesaron {total} ejecuciones de herramientas.")
    lines.append("")
    lines.append("### Hallazgos")
    for idx, run in enumerate(runs, start=1):
        name = str(run.get("name", "tool"))
        args = run.get("args", {}) or {}
        output = run.get("output")
        entity = _pick_entity_id(args, output)
        status = _pick_status(output)
        if status.startswith("error="):
            errors += 1
        detail = f" ({entity})" if entity else ""
        lines.append(f"{idx}. {name}{detail}: {status}")

    lines.append("")
    lines.append("### Conclusiones")
    if errors == 0:
        lines.append("- No se detectaron errores de ejecución en las evidencias disponibles.")
    else:
        lines.append(f"- Se detectaron {errors} ejecuciones con error; revisar detalle técnico.")
    lines.append("- La respuesta se generó únicamente a partir de salidas verificadas de herramientas.")

    answer = "\n".join(lines).strip()
    if looks_like_technical_answer(answer):
        # Safety net: keep this user-facing even if data is sparse.
        return (
            "## Resultado\n"
            "Se completó el procesamiento de la solicitud con evidencia de herramientas.\n"
            "Consulta la vista profunda para el detalle técnico por ejecución."
        )
    return answer
