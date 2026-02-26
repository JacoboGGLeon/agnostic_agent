from __future__ import annotations

import json
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
