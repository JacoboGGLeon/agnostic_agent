from __future__ import annotations

"""
Lógica principal (grafo LangGraph) del Agnostic Deep Agent 2026.

Sub-grafos actuales:
- ANALYZER  → descompone el prompt (rule-based sencillo por ahora).
- PLANNER   → usa Planner LLM (Qwen3+vLLM) para generar tool_calls.
- EXECUTOR  → ejecuta tools reales (LangChain tools).
- CATCHER   → normaliza las salidas de tools a una lista de runs.
- SUMMARIZER→ construye:
    - respuesta final en modo usuario (user_answer),
    - resumen técnico del pipeline (para vistas deep/dev).
- VALIDATOR → revisa si la respuesta parece cubrir todo lo pedido.

Notas:
- Este módulo sigue usando TypedDict; todavía no está cableado
  a los modelos Pydantic de `schemas.py`.
- Ya integra memoria y kb_names en el planner, y deja
  dev_out / deep_out / user_out en el estado.
- Está pensado para casos donde el agente cruza:
    * una tabla de atributos (input A, p.ej. filas de contratos),
    * con tablas de contexto (input B, p.ej. parametrías y
      diccionarios de abreviaturas/definiciones),
    * y, opcionalmente, documentos (OCR de contratos) vía tools
      como semantic_search_in_csv + rerank_qwen3.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
import json
import re
import uuid
import xml.etree.ElementTree as ET

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    AnyMessage,
    SystemMessage,
)

from .capabilities import PlannerConfig, build_planner_system_message


# ─────────────────────────────────────────────
# Tipos de alto nivel para el "program state"
# ─────────────────────────────────────────────

class AnalyzerResult(TypedDict, total=False):
    input_payload: Dict[str, Any]
    propositional_logic: str
    subqueries: List[str]
    subqueries_logic: List[str]


class PlannerTrajectory(TypedDict, total=False):
    subquery: str
    description: str


class ExecutorStep(TypedDict, total=False):
    tool_call_id: str
    tool_name: str
    args: Dict[str, Any]


class SummaryDict(TypedDict, total=False):
    analyzer: str
    planner: str
    executor: str
    catcher: str
    summarizer: str
    final_answer: str


class ValidatorResult(TypedDict, total=False):
    all_covered: bool
    reasoning: str


class State(TypedDict, total=False):
    """
    Estado del grafo (versión 0.2):

    - messages: historial de LangChain Messages.
    - analyzer: resultado ligero del ANALYZER rule-based.
    - planner_trajs: trazas de planificación del PLANNER.
    - executor_steps: pasos efectivamente ejecutados (EXECUTOR).
    - tool_runs: lista de runs normalizados (CATCHER).
    - summary / pipeline_summary: SummaryDict de todo el pipeline.
    - validator: ValidatorResult simple (cobertura / razonamiento).
    - user_prompt / session_id / kb_names / memory_context:
        metadatos que llegan desde Agent (o el llamador).
    - dev_out / deep_out / user_out:
        vistas finales que el Agent puede usar directamente.
    - llm_raw_out / llm_clean_out:
        invariantes para salida directa del modelo (sin tools),
        donde llm_clean_out = llm_raw_out sin <think>...</think>.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    analyzer: Optional[AnalyzerResult]
    planner_trajs: List[PlannerTrajectory]
    executor_steps: List[ExecutorStep]
    tool_runs: List[Dict[str, Any]]
    summary: Optional[SummaryDict]
    pipeline_summary: Optional[SummaryDict]
    validator: Optional[ValidatorResult]

    # Metadatos / contexto
    user_prompt: Optional[str]
    session_id: Optional[str]
    kb_names: List[str]
    memory_context: Optional[Dict[str, Any]]

    # Vistas finales (pueden ser rellenadas por SUMMARIZER)
    dev_out: Optional[str]
    deep_out: Optional[str]
    user_out: Optional[str]

    # Invariantes de salida (para modo sin tools)
    llm_raw_out: Optional[str]
    llm_clean_out: Optional[str]


# ─────────────────────────────────────────────
# Planner runtime helpers (tool_calls)
# ─────────────────────────────────────────────

def _coerce_content_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", "") or p.get("content", "") or "")
            else:
                parts.append(str(p))
        return "".join(parts)
    return "" if content is None else str(content)


def _parse_args_maybe_json(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_toolcalls_list(raw_calls: Any) -> List[Dict[str, Any]]:
    """
    Normaliza múltiples formatos a:
      [{"id": ..., "name": ..., "args": {...}}, ...]
    """
    norm: List[Dict[str, Any]] = []
    if not raw_calls:
        return norm

    # ✅ robustez: a veces viene dict o un objeto suelto
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    elif not isinstance(raw_calls, list):
        raw_calls = [raw_calls]

    for c in raw_calls:
        if isinstance(c, dict):
            fn = c.get("function") or {}
            name = c.get("name") or fn.get("name") or c.get("tool_name")
            if "args" in c:
                args_raw = c.get("args")
            else:
                args_raw = fn.get("arguments") or c.get("arguments") or c.get("parameters")
            id_ = c.get("id") or c.get("tool_call_id")
        else:
            fn = getattr(c, "function", None)
            name = (
                getattr(c, "name", None)
                or (getattr(fn, "name", None) if fn else None)
                or getattr(c, "tool_name", None)
            )
            args_raw = (
                getattr(c, "args", None)
                or (getattr(fn, "arguments", None) if fn else None)
                or getattr(c, "arguments", None)
                or getattr(c, "parameters", None)
            )
            id_ = getattr(c, "id", None) or getattr(c, "tool_call_id", None)

        args = _parse_args_maybe_json(args_raw)
        if name:
            norm.append(
                {
                    "id": id_ or f"call_{uuid.uuid4().hex}",
                    "name": name,
                    "args": args,
                }
            )
    return norm


# ─────────────────────────────────────────────
# ✅ XML fallback robusto (Qwen XML)
# ─────────────────────────────────────────────

def _scan_balanced_json(s: str, i: int) -> Tuple[Optional[str], int]:
    """
    Escanea desde s[i] (debe ser '{') y devuelve (json_str, next_index)
    contando llaves y respetando strings/escapes.
    """
    if i < 0 or i >= len(s) or s[i] != "{":
        return None, i

    depth = 0
    in_str = False
    esc = False
    start = i

    while i < len(s):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1], i + 1
        i += 1

    return None, i


def _extract_tool_calls_via_etree(text: str) -> List[Dict[str, Any]]:
    """
    Extrae <tool_call>...</tool_call> como dicts (JSON dentro) usando XML real.
    """
    wrapped = f"<root>{text}</root>"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return []

    out: List[Dict[str, Any]] = []
    for node in root.findall(".//tool_call"):
        raw = "".join(node.itertext()).strip()
        if not raw:
            continue

        # JSON directo
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out.extend([it for it in obj if isinstance(it, dict)])
            continue
        except Exception:
            pass

        # brace-scan dentro del texto del tag
        j = raw.find("{")
        if j != -1:
            js, _ = _scan_balanced_json(raw, j)
            if js:
                try:
                    obj2 = json.loads(js)
                    if isinstance(obj2, dict):
                        out.append(obj2)
                except Exception:
                    pass

    return out


def _extract_tool_calls_via_xmlish_bracescan(text: str) -> List[Dict[str, Any]]:
    """
    Cuando el XML viene malformado, buscamos bloques <tool_call>...</tool_call>
    y adentro hacemos JSON parse o brace-scan.
    """
    out: List[Dict[str, Any]] = []
    tag_open = "<tool_call>"
    tag_close = "</tool_call>"

    pos = 0
    while True:
        a = text.find(tag_open, pos)
        if a == -1:
            break
        b = text.find(tag_close, a)
        if b == -1:
            break

        chunk = text[a + len(tag_open) : b].strip()

        # JSON directo
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out.extend([it for it in obj if isinstance(it, dict)])
        except Exception:
            # brace-scan
            j = chunk.find("{")
            if j != -1:
                js, _ = _scan_balanced_json(chunk, j)
                if js:
                    try:
                        obj2 = json.loads(js)
                        if isinstance(obj2, dict):
                            out.append(obj2)
                    except Exception:
                        pass

        pos = b + len(tag_close)

    return out


def _extract_qwen_xml_calls(ai_msg: AIMessage) -> List[Dict[str, Any]]:
    """
    Fallback robusto para Qwen3-XML:
      1) intenta XML real (ElementTree) con wrapper <root>
      2) si falla (XML roto), usa búsqueda xml-ish + brace-scan
    Luego normaliza a {"id","name","args"} (misma forma que el resto).
    """
    text = _coerce_content_str(getattr(ai_msg, "content", ""))
    if "<tool_call" not in text:
        return []

    parsed = _extract_tool_calls_via_etree(text)
    if not parsed:
        parsed = _extract_tool_calls_via_xmlish_bracescan(text)

    calls: List[Dict[str, Any]] = []
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool_name")
        args_raw = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
        args = _parse_args_maybe_json(args_raw)
        if name:
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "name": name,
                    "args": args,
                }
            )
    return calls


def extract_tool_calls(ai_msg: AIMessage) -> List[Dict[str, Any]]:
    """
    API robusta para obtener tool_calls de un AIMessage.
    Compatible con:
    - tool_calls nativos (OpenAI / qwen)
    - additional_kwargs["tool_calls"]
    - XML qwen3-XML (<tool_call>{...}</tool_call>)
    """
    if not isinstance(ai_msg, AIMessage):
        return []

    tc = getattr(ai_msg, "tool_calls", None)
    norm = _normalize_toolcalls_list(tc)
    if norm:
        return norm

    addkw = getattr(ai_msg, "additional_kwargs", {}) or {}
    tc2 = addkw.get("tool_calls")
    norm2 = _normalize_toolcalls_list(tc2)
    if norm2:
        return norm2

    return _extract_qwen_xml_calls(ai_msg)


def call_planner_with_retry(
    planner_llm,
    system_message: SystemMessage,
    user_or_history_messages: List[AnyMessage],
    planner_config: PlannerConfig,
    extra_system_messages: Optional[List[SystemMessage]] = None,
) -> AIMessage:
    """
    Llama al planner_llm con un SystemMessage fijo + historial (+ contextos extra).
    Si no produce tool_calls, reintenta hasta max_retries veces.
    """
    last_ai: AIMessage | None = None
    extra = extra_system_messages or []
    for _ in range(planner_config.max_retries + 1):
        msgs = [system_message] + extra + list(user_or_history_messages)
        ai_msg: AIMessage = planner_llm.invoke(msgs)
        last_ai = ai_msg
        if extract_tool_calls(ai_msg):
            break
    return last_ai  # type: ignore[return-value]


# ─────────────────────────────────────────────
# Helpers JSON para serializar salidas de tools
# ─────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    """
    Fallback para tipos no JSON-serializables (np.int64, sets, etc.).
    Mantiene estructura lo mejor posible en lugar de castear todo a str.
    """
    # Numpy genéricos → .item()
    try:
        import numpy as _np  # import local para no romper si no hay numpy
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass

    # Sets → lista
    if isinstance(obj, (set, frozenset)):
        return list(obj)

    # Último recurso
    return str(obj)


# ─────────────────────────────────────────────
# 1) Utilidades: strip_think() + “último assistant real”
# ─────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.S | re.I)

def strip_think(txt: str) -> str:
    """Elimina <think>...</think> si existe (Qwen / Hermes), y recorta."""
    if not isinstance(txt, str):
        return ""
    return _THINK_RE.sub("", txt).strip()

def _is_pipeline_internal_ai(m: AnyMessage) -> bool:
    """
    Detecta mensajes internos del pipeline (summarizer/validator),
    para NO confundirlos con la respuesta real del LLM.
    """
    if not isinstance(m, AIMessage):
        return False

    addkw = getattr(m, "additional_kwargs", {}) or {}
    if addkw.get("pipeline_internal") is True:
        return True

    # Heurística por contenido (fallback defensivo)
    txt = _coerce_content_str(getattr(m, "content", "")).lstrip()
    if txt.startswith("## Resumen del pipeline"):
        return True
    if txt.startswith("## Resumen deep del pipeline"):
        return True
    if txt.startswith("### VALIDATOR"):
        return True

    return False

def find_last_assistant_real(messages: List[AnyMessage]) -> Optional[AIMessage]:
    """
    Devuelve el último AIMessage "real" (del LLM), ignorando mensajes internos del pipeline.
    """
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) and not _is_pipeline_internal_ai(m):
            txt = _coerce_content_str(getattr(m, "content", "")).strip()
            if txt:
                return m
    return None


# ─────────────────────────────────────────────
# Summarizer helpers
# ─────────────────────────────────────────────

def _fmt_args(args: dict) -> str:
    if not args:
        return ""
    return ", ".join(f"{k}={repr(v)}" for k, v in args.items())


def _fmt_output(tool_name: str, v: Any) -> str:
    if isinstance(v, bool):
        return "Sí" if v else "No"

    # Embeddings → preview compacto
    if tool_name == "embed_texts":
        try:
            preview = []
            if isinstance(v, list):
                for idx, vec in enumerate(v):
                    if isinstance(vec, list):
                        preview.append(
                            {
                                "index": idx,
                                "dim": len(vec),
                                "head_5": vec[:5],
                            }
                        )
            return json.dumps(preview, ensure_ascii=False, indent=2)
        except Exception:
            return str(v)

    # Reranker → lista JSON bonita
    if tool_name == "rerank_qwen3":
        try:
            return json.dumps(v, ensure_ascii=False, indent=2)
        except Exception:
            return str(v)

    # Tablas / dicts grandes → JSON bonito
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False, indent=2, default=_json_default)
        except Exception:
            return str(v)

    return str(v)


def summarize_tool_runs(user_text: str, runs: List[Dict[str, Any]]) -> str:
    """
    Resumen user-friendly basado SOLO en las salidas de herramientas.
    Esto alimenta `summary.summarizer` y la sección dev "SUMMARIZER (basado en herramientas)".
    """
    if not runs:
        return (
            "No se invocó ninguna herramienta. "
            "No puedo responder con garantías a la pregunta sólo con razonamiento interno."
        )

    partes = [
        "📌 **Resumen basado en herramientas (sin alucinaciones)**",
    ]

    for r in runs:
        arg_str = _fmt_args(r["args"])
        out_str = _fmt_output(r["name"], r["output"])

        if r["name"] in (
            "embed_texts",
            "rerank_qwen3",
            "embed_context_tables",
            "semantic_search_in_csv",
            "judge_row_with_context",
        ):
            partes.append(
                f"- `{r['name']}({arg_str})`:\n\n```json\n{out_str}\n```"
            )
        else:
            partes.append(
                f"- `{r['name']}({arg_str})` → **{out_str}**"
            )

    return "\n".join(partes)


def build_user_answer(user_text: str, runs: List[Dict[str, Any]]) -> str:
    """
    Construye la respuesta 1:1 en lenguaje natural para el modo USER,
    usando EXCLUSIVAMENTE lo que viene en `runs` (tool-first, sin alucinaciones).

    Nota: es agnóstico, pero tiene atajos para algunas tools típicas
    (to_upper, is_palindrome, word_count, embed_texts, rerank_qwen3,
     y las tools de contexto: embed_context_tables,
     semantic_search_in_csv, judge_row_with_context).

    Si no reconoce nada, devolvemos "" y el caller
    hará fallback al resumen de herramientas.
    """
    if not runs:
        return ""

    sentences: List[str] = []

    # 1) Reranker: priorizamos porque suele ser la "respuesta clave"
    rr = next((r for r in runs if r.get("name") == "rerank_qwen3"), None)
    if rr is not None:
        out = rr.get("output")
        args = rr.get("args", {}) or {}
        docs = args.get("documents") or []

        if isinstance(out, list) and out:
            best = out[0]
            idx = best.get("index", 0)
            doc = best.get("document")
            score = best.get("score")

            # Si el documento no viene embebido, lo recuperamos de args.documents
            if doc is None and isinstance(docs, list) and isinstance(idx, int) and 0 <= idx < len(docs):
                doc = docs[idx]

            idx_h = idx + 1 if isinstance(idx, int) else idx

            if doc is not None:
                if isinstance(score, (int, float)):
                    sentences.append(
                        f"El documento más relevante es el #{idx_h}: \"{doc}\" "
                        f"(score ≈ {score:.3f})."
                    )
                else:
                    sentences.append(
                        f"El documento más relevante es el #{idx_h}: \"{doc}\"."
                    )

    # 2) to_upper
    for r in runs:
        if r.get("name") == "to_upper":
            txt = r.get("args", {}).get("text")
            out = r.get("output")
            if txt is not None and out is not None:
                sentences.append(f"He convertido \"{txt}\" a mayúsculas: **{out}**.")

    # 3) is_palindrome
    for r in runs:
        if r.get("name") == "is_palindrome":
            txt = r.get("args", {}).get("text")
            val = r.get("output")
            if txt:
                if bool(val):
                    sentences.append(f"\"{txt}\" es un palíndromo.")
                else:
                    sentences.append(f"\"{txt}\" no es un palíndromo.")

    # 4) word_count
    for r in runs:
        if r.get("name") == "word_count":
            txt = r.get("args", {}).get("text")
            n = r.get("output")
            if txt is not None and n is not None:
                sentences.append(f"El texto \"{txt}\" tiene {n} palabras.")

    # 5) embed_texts
    emb = next((r for r in runs if r.get("name") == "embed_texts"), None)
    if emb is not None:
        vecs = emb.get("output")
        n_vecs = len(vecs) if isinstance(vecs, list) else 0

        dim: Optional[int] = None
        if isinstance(vecs, list) and vecs:
            v0 = vecs[0]
            if isinstance(v0, list):  # lista cruda de floats
                dim = len(v0)
            elif isinstance(v0, dict) and "dim" in v0:
                try:
                    dim = int(v0["dim"])
                except Exception:
                    dim = None

        if n_vecs and dim:
            sentences.append(
                f"He generado embeddings de dimensión {dim} para {n_vecs} texto(s)."
            )
        elif n_vecs:
            sentences.append(
                f"He generado embeddings para {n_vecs} texto(s)."
            )

    # 6) embed_context_tables (tablas de parametrías / abreviaturas)
    for r in runs:
        if r.get("name") == "embed_context_tables":
            args = r.get("args", {}) or {}
            tables = args.get("table_paths") or args.get("tables") or []
            out = r.get("output")

            n_tables = len(tables) if isinstance(tables, list) else 0

            dim: Optional[int] = None
            if isinstance(out, dict):
                dim = out.get("embedding_dim") or out.get("dim")
                try:
                    if dim is not None:
                        dim = int(dim)
                except Exception:
                    dim = None

            if n_tables:
                if dim:
                    sentences.append(
                        f"He generado embeddings de dimensión {dim} para {n_tables} tabla(s) de contexto "
                        "(parametrías, diccionarios o definiciones)."
                    )
                else:
                    sentences.append(
                        f"He generado embeddings para {n_tables} tabla(s) de contexto "
                        "(parametrías, diccionarios o definiciones)."
                    )

    # 7) semantic_search_in_csv (búsqueda semántica en tablas / OCR tabular)
    for r in runs:
        if r.get("name") == "semantic_search_in_csv":
            args = r.get("args", {}) or {}
            query = args.get("query")
            csv_path = args.get("csv_path") or args.get("table_path")
            out = r.get("output")

            best_row = None
            if isinstance(out, list) and out:
                best_row = out[0]

            if best_row is not None:
                score = (
                    best_row.get("score")
                    if isinstance(best_row, dict)
                    else None
                )
                row_preview = best_row.get("row") if isinstance(best_row, dict) else None
                if isinstance(row_preview, dict):
                    row_preview = json.dumps(row_preview, ensure_ascii=False)
                if query and csv_path:
                    if isinstance(score, (int, float)):
                        sentences.append(
                            f"En la tabla `{csv_path}` he encontrado una fila muy relevante para la consulta "
                            f"\"{query}\" (score ≈ {score:.3f}). Fila ejemplo: {row_preview}."
                        )
                    else:
                        sentences.append(
                            f"En la tabla `{csv_path}` he encontrado una fila relevante para la consulta "
                            f"\"{query}\". Fila ejemplo: {row_preview}."
                        )

    # 8) judge_row_with_context (juicio de una fila de atributos vs parametrías/diccionarios)
    for r in runs:
        if r.get("name") == "judge_row_with_context":
            args = r.get("args", {}) or {}
            out = r.get("output") or {}

            contract_id = (
                out.get("contract_id")
                or out.get("row_id")
                or out.get("id")
                or args.get("contract_id")
                or args.get("row_id")
                or args.get("id")
            )
            judgement = (
                out.get("judgement")
                or out.get("verdict")
                or out.get("decision")
                or out.get("status")
            )
            reasons = out.get("reasons") or out.get("rule_hits") or out.get("details")

            if isinstance(reasons, list):
                reasons_str = "; ".join(map(str, reasons[:3]))
            else:
                reasons_str = str(reasons) if reasons is not None else ""

            if contract_id is not None and judgement is not None:
                sent = (
                    f"He evaluado la fila/contrato '{contract_id}' aplicando las tablas de parametrías "
                    f"y diccionarios de contexto. El juicio es: **{judgement}**."
                )
                if reasons_str:
                    sent += f" Motivos principales: {reasons_str}."
                sentences.append(sent)
            elif judgement is not None:
                sent = (
                    f"He evaluado la fila de atributos con las tablas de contexto; "
                    f"el juicio global es: **{judgement}**."
                )
                if reasons_str:
                    sent += f" Motivos principales: {reasons_str}."
                sentences.append(sent)

    if not sentences:
        # Fallback: si por lo que sea no pudimos mapear nada,
        # devolvemos cadena vacía y el caller decidirá usar el resumen de tools.
        return ""

    return " ".join(sentences)


# ─────────────────────────────────────────────
# Pequeños helpers de contexto (memoria / KB)
# ─────────────────────────────────────────────

def _format_memory_context(mem: Any) -> str:
    """
    Serializa el memory_context para pasarlo al planner como SystemMessage.

    - Si tiene resumen conversacional, o items relevantes, se formatean aquí.
    """
    if not mem:
        return ""

    lines = []
    if isinstance(mem, dict):
        if "summary" in mem:
            lines.append(f"Resumen previo: {mem['summary']}")
        if "facts" in mem and isinstance(mem["facts"], list):
            lines.append("Hechos conocidos:")
            for f in mem["facts"]:
                lines.append(f"- {f}")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────
# LOGIC NODES (Analyzer, Planner, Executor, Catcher, Summarizer, Validator)
# ─────────────────────────────────────────────


#@title agnostic_agent/logic.py
%%writefile agnostic_agent/logic.py
from __future__ import annotations

"""
Lógica principal (grafo LangGraph) del Agnostic Deep Agent.

Sub-grafos actuales:
- ANALYZER  → descompone el prompt (rule-based sencillo por ahora).
- PLANNER   → usa Planner LLM (Qwen3+vLLM) para generar tool_calls.
- EXECUTOR  → ejecuta tools reales (LangChain tools).
- CATCHER   → normaliza las salidas de tools a una lista de runs.
- SUMMARIZER→ construye:
    - respuesta final en modo usuario (user_answer),
    - resumen técnico del pipeline (para vistas deep/dev).
- VALIDATOR → revisa si la respuesta parece cubrir todo lo pedido.

Notas:
- Este módulo sigue usando TypedDict; todavía no está cableado
  a los modelos Pydantic de `schemas.py`.
- Ya integra memoria y kb_names en el planner, y deja
  dev_out / deep_out / user_out en el estado.
- Está pensado para casos donde el agente cruza:
    * una tabla de atributos (input A, p.ej. filas de contratos),
    * con tablas de contexto (input B, p.ej. parametrías y
      diccionarios de abreviaturas/definiciones),
    * y, opcionalmente, documentos (OCR de contratos) vía tools
      como semantic_search_in_csv + rerank_qwen3.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
import json
import re
import uuid
import xml.etree.ElementTree as ET

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    AnyMessage,
    SystemMessage,
)

from .capabilities import PlannerConfig, build_planner_system_message


# ─────────────────────────────────────────────
# Tipos de alto nivel para el "program state"
# ─────────────────────────────────────────────

class AnalyzerResult(TypedDict, total=False):
    input_payload: Dict[str, Any]
    propositional_logic: str
    subqueries: List[str]
    subqueries_logic: List[str]


class PlannerTrajectory(TypedDict, total=False):
    subquery: str
    description: str


class ExecutorStep(TypedDict, total=False):
    tool_call_id: str
    tool_name: str
    args: Dict[str, Any]


class SummaryDict(TypedDict, total=False):
    analyzer: str
    planner: str
    executor: str
    catcher: str
    summarizer: str
    final_answer: str


class ValidatorResult(TypedDict, total=False):
    all_covered: bool
    reasoning: str


class State(TypedDict, total=False):
    """
    Estado del grafo (versión 0.2):

    - messages: historial de LangChain Messages.
    - analyzer: resultado ligero del ANALYZER rule-based.
    - planner_trajs: trazas de planificación del PLANNER.
    - executor_steps: pasos efectivamente ejecutados (EXECUTOR).
    - tool_runs: lista de runs normalizados (CATCHER).
    - summary / pipeline_summary: SummaryDict de todo el pipeline.
    - validator: ValidatorResult simple (cobertura / razonamiento).
    - user_prompt / session_id / kb_names / memory_context:
        metadatos que llegan desde Agent (o el llamador).
    - dev_out / deep_out / user_out:
        vistas finales que el Agent puede usar directamente.
    - llm_raw_out / llm_clean_out:
        invariantes para salida directa del modelo (sin tools),
        donde llm_clean_out = llm_raw_out sin <think>...</think>.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    analyzer: Optional[AnalyzerResult]
    planner_trajs: List[PlannerTrajectory]
    executor_steps: List[ExecutorStep]
    tool_runs: List[Dict[str, Any]]
    summary: Optional[SummaryDict]
    pipeline_summary: Optional[SummaryDict]
    validator: Optional[ValidatorResult]

    # Metadatos / contexto
    user_prompt: Optional[str]
    session_id: Optional[str]
    kb_names: List[str]
    memory_context: Optional[Dict[str, Any]]

    # Vistas finales (pueden ser rellenadas por SUMMARIZER)
    dev_out: Optional[str]
    deep_out: Optional[str]
    user_out: Optional[str]

    # Invariantes de salida (para modo sin tools)
    llm_raw_out: Optional[str]
    llm_clean_out: Optional[str]


# ─────────────────────────────────────────────
# Planner runtime helpers (tool_calls)
# ─────────────────────────────────────────────

def _coerce_content_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", "") or p.get("content", "") or "")
            else:
                parts.append(str(p))
        return "".join(parts)
    return "" if content is None else str(content)


def _parse_args_maybe_json(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_toolcalls_list(raw_calls: Any) -> List[Dict[str, Any]]:
    """
    Normaliza múltiples formatos a:
      [{"id": ..., "name": ..., "args": {...}}, ...]
    """
    norm: List[Dict[str, Any]] = []
    if not raw_calls:
        return norm

    # ✅ robustez: a veces viene dict o un objeto suelto
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    elif not isinstance(raw_calls, list):
        raw_calls = [raw_calls]

    for c in raw_calls:
        if isinstance(c, dict):
            fn = c.get("function") or {}
            name = c.get("name") or fn.get("name") or c.get("tool_name")
            if "args" in c:
                args_raw = c.get("args")
            else:
                args_raw = fn.get("arguments") or c.get("arguments") or c.get("parameters")
            id_ = c.get("id") or c.get("tool_call_id")
        else:
            fn = getattr(c, "function", None)
            name = (
                getattr(c, "name", None)
                or (getattr(fn, "name", None) if fn else None)
                or getattr(c, "tool_name", None)
            )
            args_raw = (
                getattr(c, "args", None)
                or (getattr(fn, "arguments", None) if fn else None)
                or getattr(c, "arguments", None)
                or getattr(c, "parameters", None)
            )
            id_ = getattr(c, "id", None) or getattr(c, "tool_call_id", None)

        args = _parse_args_maybe_json(args_raw)
        if name:
            norm.append(
                {
                    "id": id_ or f"call_{uuid.uuid4().hex}",
                    "name": name,
                    "args": args,
                }
            )
    return norm


# ─────────────────────────────────────────────
# ✅ XML fallback robusto (Qwen XML)
# ─────────────────────────────────────────────

def _scan_balanced_json(s: str, i: int) -> Tuple[Optional[str], int]:
    """
    Escanea desde s[i] (debe ser '{') y devuelve (json_str, next_index)
    contando llaves y respetando strings/escapes.
    """
    if i < 0 or i >= len(s) or s[i] != "{":
        return None, i

    depth = 0
    in_str = False
    esc = False
    start = i

    while i < len(s):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1], i + 1
        i += 1

    return None, i


def _extract_tool_calls_via_etree(text: str) -> List[Dict[str, Any]]:
    """
    Extrae <tool_call>...</tool_call> como dicts (JSON dentro) usando XML real.
    """
    wrapped = f"<root>{text}</root>"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return []

    out: List[Dict[str, Any]] = []
    for node in root.findall(".//tool_call"):
        raw = "".join(node.itertext()).strip()
        if not raw:
            continue

        # JSON directo
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out.extend([it for it in obj if isinstance(it, dict)])
            continue
        except Exception:
            pass

        # brace-scan dentro del texto del tag
        j = raw.find("{")
        if j != -1:
            js, _ = _scan_balanced_json(raw, j)
            if js:
                try:
                    obj2 = json.loads(js)
                    if isinstance(obj2, dict):
                        out.append(obj2)
                except Exception:
                    pass

    return out


def _extract_tool_calls_via_xmlish_bracescan(text: str) -> List[Dict[str, Any]]:
    """
    Cuando el XML viene malformado, buscamos bloques <tool_call>...</tool_call>
    y adentro hacemos JSON parse o brace-scan.
    """
    out: List[Dict[str, Any]] = []
    tag_open = "<tool_call>"
    tag_close = "</tool_call>"

    pos = 0
    while True:
        a = text.find(tag_open, pos)
        if a == -1:
            break
        b = text.find(tag_close, a)
        if b == -1:
            break

        chunk = text[a + len(tag_open) : b].strip()

        # JSON directo
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out.extend([it for it in obj if isinstance(it, dict)])
        except Exception:
            # brace-scan
            j = chunk.find("{")
            if j != -1:
                js, _ = _scan_balanced_json(chunk, j)
                if js:
                    try:
                        obj2 = json.loads(js)
                        if isinstance(obj2, dict):
                            out.append(obj2)
                    except Exception:
                        pass

        pos = b + len(tag_close)

    return out


def _extract_qwen_xml_calls(ai_msg: AIMessage) -> List[Dict[str, Any]]:
    """
    Fallback robusto para Qwen3-XML:
      1) intenta XML real (ElementTree) con wrapper <root>
      2) si falla (XML roto), usa búsqueda xml-ish + brace-scan
    Luego normaliza a {"id","name","args"} (misma forma que el resto).
    """
    text = _coerce_content_str(getattr(ai_msg, "content", ""))
    if "<tool_call" not in text:
        return []

    parsed = _extract_tool_calls_via_etree(text)
    if not parsed:
        parsed = _extract_tool_calls_via_xmlish_bracescan(text)

    calls: List[Dict[str, Any]] = []
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool_name")
        args_raw = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
        args = _parse_args_maybe_json(args_raw)
        if name:
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "name": name,
                    "args": args,
                }
            )
    return calls


def extract_tool_calls(ai_msg: AIMessage) -> List[Dict[str, Any]]:
    """
    API robusta para obtener tool_calls de un AIMessage.
    Compatible con:
    - tool_calls nativos (OpenAI / qwen)
    - additional_kwargs["tool_calls"]
    - XML qwen3-XML (<tool_call>{...}</tool_call>)
    """
    if not isinstance(ai_msg, AIMessage):
        return []

    tc = getattr(ai_msg, "tool_calls", None)
    norm = _normalize_toolcalls_list(tc)
    if norm:
        return norm

    addkw = getattr(ai_msg, "additional_kwargs", {}) or {}
    tc2 = addkw.get("tool_calls")
    norm2 = _normalize_toolcalls_list(tc2)
    if norm2:
        return norm2

    return _extract_qwen_xml_calls(ai_msg)


def call_planner_with_retry(
    planner_llm,
    system_message: SystemMessage,
    user_or_history_messages: List[AnyMessage],
    planner_config: PlannerConfig,
    extra_system_messages: Optional[List[SystemMessage]] = None,
) -> AIMessage:
    """
    Llama al planner_llm con un SystemMessage fijo + historial (+ contextos extra).
    Si no produce tool_calls, reintenta hasta max_retries veces.
    """
    last_ai: AIMessage | None = None
    extra = extra_system_messages or []
    for _ in range(planner_config.max_retries + 1):
        msgs = [system_message] + extra + list(user_or_history_messages)
        ai_msg: AIMessage = planner_llm.invoke(msgs)
        last_ai = ai_msg
        if extract_tool_calls(ai_msg):
            break
    return last_ai  # type: ignore[return-value]


# ─────────────────────────────────────────────
# Helpers JSON para serializar salidas de tools
# ─────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    """
    Fallback para tipos no JSON-serializables (np.int64, sets, etc.).
    Mantiene estructura lo mejor posible en lugar de castear todo a str.
    """
    # Numpy genéricos → .item()
    try:
        import numpy as _np  # import local para no romper si no hay numpy
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass

    # Sets → lista
    if isinstance(obj, (set, frozenset)):
        return list(obj)

    # Último recurso
    return str(obj)


# ─────────────────────────────────────────────
# 1) Utilidades: strip_think() + “último assistant real”
# ─────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.S | re.I)

def strip_think(txt: str) -> str:
    """Elimina <think>...</think> si existe (Qwen / Hermes), y recorta."""
    if not isinstance(txt, str):
        return ""
    return _THINK_RE.sub("", txt).strip()

def _is_pipeline_internal_ai(m: AnyMessage) -> bool:
    """
    Detecta mensajes internos del pipeline (summarizer/validator),
    para NO confundirlos con la respuesta real del LLM.
    """
    if not isinstance(m, AIMessage):
        return False

    addkw = getattr(m, "additional_kwargs", {}) or {}
    if addkw.get("pipeline_internal") is True:
        return True

    # Heurística por contenido (fallback defensivo)
    txt = _coerce_content_str(getattr(m, "content", "")).lstrip()
    if txt.startswith("## Resumen del pipeline"):
        return True
    if txt.startswith("## Resumen deep del pipeline"):
        return True
    if txt.startswith("### VALIDATOR"):
        return True

    return False

def find_last_assistant_real(messages: List[AnyMessage]) -> Optional[AIMessage]:
    """
    Devuelve el último AIMessage "real" (del LLM), ignorando mensajes internos del pipeline.
    """
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) and not _is_pipeline_internal_ai(m):
            txt = _coerce_content_str(getattr(m, "content", "")).strip()
            if txt:
                return m
    return None


# ─────────────────────────────────────────────
# Summarizer helpers
# ─────────────────────────────────────────────

def _fmt_args(args: dict) -> str:
    if not args:
        return ""
    return ", ".join(f"{k}={repr(v)}" for k, v in args.items())


def _fmt_output(tool_name: str, v: Any) -> str:
    if isinstance(v, bool):
        return "Sí" if v else "No"

    # Embeddings → preview compacto
    if tool_name == "embed_texts":
        try:
            preview = []
            if isinstance(v, list):
                for idx, vec in enumerate(v):
                    if isinstance(vec, list):
                        preview.append(
                            {
                                "index": idx,
                                "dim": len(vec),
                                "head_5": vec[:5],
                            }
                        )
            return json.dumps(preview, ensure_ascii=False, indent=2)
        except Exception:
            return str(v)

    # Reranker → lista JSON bonita
    if tool_name == "rerank_qwen3":
        try:
            return json.dumps(v, ensure_ascii=False, indent=2)
        except Exception:
            return str(v)

    # Tablas / dicts grandes → JSON bonito
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False, indent=2, default=_json_default)
        except Exception:
            return str(v)

    return str(v)


def summarize_tool_runs(user_text: str, runs: List[Dict[str, Any]]) -> str:
    """
    Resumen user-friendly basado SOLO en las salidas de herramientas.
    Esto alimenta `summary.summarizer` y la sección dev "SUMMARIZER (basado en herramientas)".
    """
    if not runs:
        return (
            "No se invocó ninguna herramienta. "
            "No puedo responder con garantías a la pregunta sólo con razonamiento interno."
        )

    partes = [
        "📌 **Resumen basado en herramientas (sin alucinaciones)**",
    ]

    for r in runs:
        arg_str = _fmt_args(r["args"])
        out_str = _fmt_output(r["name"], r["output"])

        if r["name"] in (
            "embed_texts",
            "rerank_qwen3",
            "embed_context_tables",
            "semantic_search_in_csv",
            "judge_row_with_context",
        ):
            partes.append(
                f"- `{r['name']}({arg_str})`:\n\n```json\n{out_str}\n```"
            )
        else:
            partes.append(
                f"- `{r['name']}({arg_str})` → **{out_str}**"
            )

    return "\n".join(partes)


def build_user_answer(user_text: str, runs: List[Dict[str, Any]]) -> str:
    """
    Construye la respuesta 1:1 en lenguaje natural para el modo USER,
    usando EXCLUSIVAMENTE lo que viene en `runs` (tool-first, sin alucinaciones).

    Nota: es agnóstico, pero tiene atajos para algunas tools típicas
    (to_upper, is_palindrome, word_count, embed_texts, rerank_qwen3,
     y las tools de contexto: embed_context_tables,
     semantic_search_in_csv, judge_row_with_context).

    Si no reconoce nada, devolvemos "" y el caller
    hará fallback al resumen de herramientas.
    """
    if not runs:
        return ""

    sentences: List[str] = []

    # 1) Reranker: priorizamos porque suele ser la "respuesta clave"
    rr = next((r for r in runs if r.get("name") == "rerank_qwen3"), None)
    if rr is not None:
        out = rr.get("output")
        args = rr.get("args", {}) or {}
        docs = args.get("documents") or []

        if isinstance(out, list) and out:
            best = out[0]
            idx = best.get("index", 0)
            doc = best.get("document")
            score = best.get("score")

            # Si el documento no viene embebido, lo recuperamos de args.documents
            if doc is None and isinstance(docs, list) and isinstance(idx, int) and 0 <= idx < len(docs):
                doc = docs[idx]

            idx_h = idx + 1 if isinstance(idx, int) else idx

            if doc is not None:
                if isinstance(score, (int, float)):
                    sentences.append(
                        f"El documento más relevante es el #{idx_h}: \"{doc}\" "
                        f"(score ≈ {score:.3f})."
                    )
                else:
                    sentences.append(
                        f"El documento más relevante es el #{idx_h}: \"{doc}\"."
                    )

    # 2) to_upper
    for r in runs:
        if r.get("name") == "to_upper":
            txt = r.get("args", {}).get("text")
            out = r.get("output")
            if txt is not None and out is not None:
                sentences.append(f"He convertido \"{txt}\" a mayúsculas: **{out}**.")

    # 3) is_palindrome
    for r in runs:
        if r.get("name") == "is_palindrome":
            txt = r.get("args", {}).get("text")
            val = r.get("output")
            if txt:
                if bool(val):
                    sentences.append(f"\"{txt}\" es un palíndromo.")
                else:
                    sentences.append(f"\"{txt}\" no es un palíndromo.")

    # 4) word_count
    for r in runs:
        if r.get("name") == "word_count":
            txt = r.get("args", {}).get("text")
            n = r.get("output")
            if txt is not None and n is not None:
                sentences.append(f"El texto \"{txt}\" tiene {n} palabras.")

    # 5) embed_texts
    emb = next((r for r in runs if r.get("name") == "embed_texts"), None)
    if emb is not None:
        vecs = emb.get("output")
        n_vecs = len(vecs) if isinstance(vecs, list) else 0

        dim: Optional[int] = None
        if isinstance(vecs, list) and vecs:
            v0 = vecs[0]
            if isinstance(v0, list):  # lista cruda de floats
                dim = len(v0)
            elif isinstance(v0, dict) and "dim" in v0:
                try:
                    dim = int(v0["dim"])
                except Exception:
                    dim = None

        if n_vecs and dim:
            sentences.append(
                f"He generado embeddings de dimensión {dim} para {n_vecs} texto(s)."
            )
        elif n_vecs:
            sentences.append(
                f"He generado embeddings para {n_vecs} texto(s)."
            )

    # 6) embed_context_tables (tablas de parametrías / abreviaturas)
    for r in runs:
        if r.get("name") == "embed_context_tables":
            args = r.get("args", {}) or {}
            tables = args.get("table_paths") or args.get("tables") or []
            out = r.get("output")

            n_tables = len(tables) if isinstance(tables, list) else 0

            dim: Optional[int] = None
            if isinstance(out, dict):
                dim = out.get("embedding_dim") or out.get("dim")
                try:
                    if dim is not None:
                        dim = int(dim)
                except Exception:
                    dim = None

            if n_tables:
                if dim:
                    sentences.append(
                        f"He generado embeddings de dimensión {dim} para {n_tables} tabla(s) de contexto "
                        "(parametrías, diccionarios o definiciones)."
                    )
                else:
                    sentences.append(
                        f"He generado embeddings para {n_tables} tabla(s) de contexto "
                        "(parametrías, diccionarios o definiciones)."
                    )

    # 7) semantic_search_in_csv (búsqueda semántica en tablas / OCR tabular)
    for r in runs:
        if r.get("name") == "semantic_search_in_csv":
            args = r.get("args", {}) or {}
            query = args.get("query")
            csv_path = args.get("csv_path") or args.get("table_path")
            out = r.get("output")

            best_row = None
            if isinstance(out, list) and out:
                best_row = out[0]

            if best_row is not None:
                score = (
                    best_row.get("score")
                    if isinstance(best_row, dict)
                    else None
                )
                row_preview = best_row.get("row") if isinstance(best_row, dict) else None
                if isinstance(row_preview, dict):
                    row_preview = json.dumps(row_preview, ensure_ascii=False)
                if query and csv_path:
                    if isinstance(score, (int, float)):
                        sentences.append(
                            f"En la tabla `{csv_path}` he encontrado una fila muy relevante para la consulta "
                            f"\"{query}\" (score ≈ {score:.3f}). Fila ejemplo: {row_preview}."
                        )
                    else:
                        sentences.append(
                            f"En la tabla `{csv_path}` he encontrado una fila relevante para la consulta "
                            f"\"{query}\". Fila ejemplo: {row_preview}."
                        )

    # 8) judge_row_with_context (juicio de una fila de atributos vs parametrías/diccionarios)
    for r in runs:
        if r.get("name") == "judge_row_with_context":
            args = r.get("args", {}) or {}
            out = r.get("output") or {}

            contract_id = (
                out.get("contract_id")
                or out.get("row_id")
                or out.get("id")
                or args.get("contract_id")
                or args.get("row_id")
                or args.get("id")
            )
            judgement = (
                out.get("judgement")
                or out.get("verdict")
                or out.get("decision")
                or out.get("status")
            )
            reasons = out.get("reasons") or out.get("rule_hits") or out.get("details")

            if isinstance(reasons, list):
                reasons_str = "; ".join(map(str, reasons[:3]))
            else:
                reasons_str = str(reasons) if reasons is not None else ""

            if contract_id is not None and judgement is not None:
                sent = (
                    f"He evaluado la fila/contrato '{contract_id}' aplicando las tablas de parametrías "
                    f"y diccionarios de contexto. El juicio es: **{judgement}**."
                )
                if reasons_str:
                    sent += f" Motivos principales: {reasons_str}."
                sentences.append(sent)
            elif judgement is not None:
                sent = (
                    f"He evaluado la fila de atributos con las tablas de contexto; "
                    f"el juicio global es: **{judgement}**."
                )
                if reasons_str:
                    sent += f" Motivos principales: {reasons_str}."
                sentences.append(sent)

    if not sentences:
        # Fallback: si por lo que sea no pudimos mapear nada,
        # devolvemos cadena vacía y el caller decidirá usar el resumen de tools.
        return ""

    return " ".join(sentences)


# ─────────────────────────────────────────────
# Pequeños helpers de contexto (memoria / KB)
# ─────────────────────────────────────────────

def _format_memory_context(mem: Any) -> str:
    """
    Serializa el memory_context para pasarlo al planner como SystemMessage.

    Pensado para cosas tipo:
      - últimas N interacciones relevantes,
      - notas de usuario,
      - resúmenes de largo plazo.

    Mantenerlo breve es trabajo de memory.py; aquí sólo lo volcamos.
    """
    if not mem:
        return ""
    try:
        return json.dumps(mem, ensure_ascii=False, indent=2)
    except Exception:
        return str(mem)


def _format_kb_hint(kb_names: List[str]) -> str:
    if not kb_names:
        return ""
    return (
        "KBs disponibles para esta sesión:\n"
        + "\n".join(f"- {name}" for name in kb_names)
        + "\n\nPuedes decidir llamar a herramientas que lean o crucen estas KBs "
          "si es necesario (por ejemplo, comparar filas de una tabla con una tabla "
          "de parámetros / reglas de calidad y emitir una tabla de juicios)."
    )


# ─────────────────────────────────────────────
# Builder del grafo LangGraph
# ─────────────────────────────────────────────

def build_graph_agent(
    planner_llm,
    tools: List[Any],
    planner_config: PlannerConfig | None = None,
):
    """
    Grafo:

        START → ANALYZER → PLANNER
                      ├─(tool_calls)→ EXECUTOR → CATCHER → SUMMARIZER → VALIDATOR → END
                      └─────────────→ SUMMARIZER → VALIDATOR → END
    """
    cfg = planner_config or PlannerConfig()
    system_msg = build_planner_system_message(cfg)

    # ANALYZER (rule-based inicial, pero ya guarda payload rico)
    def analyzer_node(state: State) -> Dict[str, Any]:
        messages = state.get("messages", [])
        user_messages = [m for m in messages if isinstance(m, HumanMessage)]
        last_user = user_messages[-1] if user_messages else None
        user_text = last_user.content if isinstance(last_user, HumanMessage) else ""

        # Permitimos que el llamador ya haya rellenado user_prompt
        user_prompt = state.get("user_prompt") or user_text

        # Input payload más rico: aquí podríamos meter referencias a tablas/JSON.
        input_payload: Dict[str, Any] = {
            "user_prompt": user_prompt,
        }

        # Split sencillo multi-sentencia: puntos y saltos de línea
        raw = str(user_prompt).replace("\n", " ")
        subqueries: List[str] = []
        for part in raw.split("."):
            part = part.strip()
            if part:
                subqueries.append(part)
        if not subqueries and user_prompt:
            subqueries = [user_prompt]

        subqueries_logic = [f"q{i+1}" for i in range(len(subqueries))]
        propositional_logic = " ∧ ".join(subqueries_logic) if subqueries_logic else ""

        analyzer: AnalyzerResult = {
            "input_payload": input_payload,
            "propositional_logic": propositional_logic,
            "subqueries": subqueries,
            "subqueries_logic": subqueries_logic,
        }

        return {"analyzer": analyzer}

    # PLANNER (ve memoria y KB names como contextos)
    def planner_node(state: State) -> Dict[str, Any]:
        msgs: List[AnyMessage] = state["messages"]

        # Contextos adicionales
        mem_ctx = state.get("memory_context")
        kb_names = state.get("kb_names") or []

        mem_str = _format_memory_context(mem_ctx)
        kb_str = _format_kb_hint(kb_names)

        extra_system_messages: List[SystemMessage] = []
        if mem_str:
            extra_system_messages.append(
                SystemMessage(
                    content=(
                        "Contexto de memoria para esta sesión "
                        "(puede contener preferencias, interacciones previas o notas):\n"
                        f"{mem_str}"
                    )
                )
            )
        if kb_str:
            extra_system_messages.append(SystemMessage(content=kb_str))

        ai_msg: AIMessage = call_planner_with_retry(
            planner_llm=planner_llm,
            system_message=system_msg,
            user_or_history_messages=msgs,
            planner_config=cfg,
            extra_system_messages=extra_system_messages,
        )

        # 5) Invariante: guardamos raw/clean del LLM (sirve para modo sin tools)
        llm_raw_out = _coerce_content_str(getattr(ai_msg, "content", ""))
        llm_clean_out = strip_think(llm_raw_out)

        tool_calls = extract_tool_calls(ai_msg)
        analyzer = state.get("analyzer") or {}
        subqs: List[str] = analyzer.get("subqueries") or []

        if not subqs:
            user_messages = [m for m in msgs if isinstance(m, HumanMessage)]
            last_user = user_messages[-1] if user_messages else None
            if isinstance(last_user, HumanMessage):
                subqs = [last_user.content]

        plan_trajs: List[PlannerTrajectory] = []
        if subqs:
            desc_lines: List[str] = []
            if not tool_calls:
                desc_lines.append(
                    "No se planificó ninguna llamada a herramientas; "
                    "el agente responderá directamente."
                )
            else:
                for idx, tc in enumerate(tool_calls, start=1):
                    desc_lines.append(
                        f"Paso {idx}: llamar a la herramienta `{tc['name']}` "
                        f"con args={tc.get('args', {})}."
                    )
            plan_trajs.append(
                PlannerTrajectory(
                    subquery=subqs[0],
                    description="\n".join(desc_lines),
                )
            )

        return {
            "messages": [ai_msg],
            "planner_trajs": plan_trajs,
            "llm_raw_out": llm_raw_out,
            "llm_clean_out": llm_clean_out,
        }

    # EXECUTOR
    def executor_node(state: State) -> Dict[str, Any]:
        messages = state["messages"]
        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
        if not ai_msgs:
            return {"messages": [], "executor_steps": []}

        ai_plan = ai_msgs[-1]
        tool_calls = extract_tool_calls(ai_plan)
        if not tool_calls:
            return {"messages": [], "executor_steps": []}

        tool_msgs: List[ToolMessage] = []
        exec_steps: List[ExecutorStep] = []

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {}) or {}

            try:
                tool_obj = next(t for t in tools if t.name == name)
                observation = tool_obj.invoke(args)
            except StopIteration:
                observation = {"error": f"Tool '{name}' no encontrada."}
            except Exception as e:
                observation = {"error": f"Excepción ejecutando tool '{name}': {e!r}"}

            try:
                payload = json.dumps(
                    {"value": observation},
                    ensure_ascii=False,
                    default=_json_default,
                )
            except TypeError:
                payload = json.dumps(
                    {"value": str(observation)},
                    ensure_ascii=False,
                )

            tool_call_id = tc["id"]

            tool_msgs.append(
                ToolMessage(
                    content=payload,
                    tool_call_id=tool_call_id,
                )
            )

            exec_steps.append(
                ExecutorStep(
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    args=args,
                )
            )

        return {
            "messages": tool_msgs,
            "executor_steps": exec_steps,
        }

    # CATCHER
    def catcher_node(state: State) -> Dict[str, Any]:
        messages = state["messages"]

        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
        ai_plan = next(
            (m for m in reversed(ai_msgs) if extract_tool_calls(m)),
            None,
        )
        tool_calls = extract_tool_calls(ai_plan) if ai_plan else []

        tmsgs: List[ToolMessage] = [m for m in messages if isinstance(m, ToolMessage)]

        runs: List[Dict[str, Any]] = []
        for tc in tool_calls:
            tm = next((t for t in tmsgs if t.tool_call_id == tc["id"]), None)
            if tm is None:
                continue
            raw = tm.content
            try:
                decoded = json.loads(raw)
                output = decoded.get("value", decoded)
            except Exception:
                output = raw
            runs.append(
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": tc.get("args", {}) or {},
                    "output": output,
                }
            )

        return {"tool_runs": runs}

    # SUMMARIZER
    def summarizer_node(state: State) -> Dict[str, Any]:
        messages = state["messages"]
        user_messages = [m for m in messages if isinstance(m, HumanMessage)]
        last_user = user_messages[-1] if user_messages else None
        user_text = last_user.content if isinstance(last_user, HumanMessage) else ""
        user_prompt = state.get("user_prompt") or user_text

        runs = state.get("tool_runs", []) or []

        # 2) Parche SUMMARIZER (regla de oro):
        # Si NO hay tools (runs vacío) y el último AI NO tiene tool_calls,
        # user_out debe ser la salida directa del LLM (limpia de <think>).
        if not runs:
            last_ai = find_last_assistant_real(messages)
            last_ai_has_tools = bool(extract_tool_calls(last_ai)) if last_ai else False

            llm_raw = state.get("llm_raw_out") or (_coerce_content_str(getattr(last_ai, "content", "")) if last_ai else "")
            llm_clean = state.get("llm_clean_out") or strip_think(llm_raw)

            if last_ai_has_tools:
                # Caso raro: se planearon tools pero no hay runs (falló executor/catcher).
                user_answer = (
                    "Se planificaron llamadas a herramientas, pero no se obtuvo ninguna salida. "
                    "Revisa EXECUTOR/CATCHER o el registro de tools."
                )
            else:
                user_answer = llm_clean or "¿Qué te gustaría hacer?"

            tools_summary_text = summarize_tool_runs(user_prompt, runs)

            analyzer = state.get("analyzer") or {}
            subqs = analyzer.get("subqueries") or []
            logic = analyzer.get("propositional_logic") or ""
            input_payload = analyzer.get("input_payload") or {}

            if analyzer:
                analyzer_text_lines = [
                    f"Input payload: {input_payload!r}",
                    f"Lógica proposicional: {logic or '(no construida)'}",
                    f"Subconsultas ({len(subqs)}):",
                ]
                for s in subqs:
                    analyzer_text_lines.append(f"- {s}")
                analyzer_text = "\n".join(analyzer_text_lines)
            else:
                analyzer_text = "No se ejecutó ANALYZER o no dejó estado."

            planner_trajs = state.get("planner_trajs", []) or []
            if planner_trajs:
                pl_lines: List[str] = []
                for i, tr in enumerate(planner_trajs, start=1):
                    pl_lines.append(f"Subquery {i}: {tr.get('subquery', '')}")
                    desc = tr.get("description")
                    if desc:
                        pl_lines.append(desc)
                planner_text = "\n".join(pl_lines)
            else:
                planner_text = (
                    "No se construyó un plan de herramientas; probablemente se respondió "
                    "directamente (o no hubo tool_calls)."
                )

            executor_steps = state.get("executor_steps", []) or []
            if executor_steps:
                ex_lines: List[str] = [
                    f"Se ejecutaron {len(executor_steps)} llamadas a herramientas:"
                ]
                for step in executor_steps:
                    ex_lines.append(
                        f"- tool_call_id={step['tool_call_id']}, "
                        f"name={step['tool_name']}, args={step['args']!r}"
                    )
                executor_text = "\n".join(ex_lines)
            else:
                executor_text = "No se ejecutó ninguna herramienta para esta consulta."

            catcher_text = "Catcher no encontró resultados de tools (runs vacío)."

            # summarizer_text para DEV/DEEP: en modo sin tools, dejamos constancia
            summarizer_text = "No se invocaron herramientas. Respuesta directa del modelo (passthrough)."

            summary_dict: SummaryDict = SummaryDict(
                analyzer=analyzer_text,
                planner=planner_text,
                executor=executor_text,
                catcher=catcher_text,
                summarizer=summarizer_text,
                final_answer=user_answer,
            )

            sections = [
                "## Resumen del pipeline",
                "### ANALYZER",
                analyzer_text,
                "### PLANNER",
                planner_text,
                "### EXECUTOR",
                executor_text,
                "### CATCHER",
                catcher_text,
                "### SUMMARIZER (basado en herramientas)",
                tools_summary_text,
                "### RESPUESTA FINAL (modo usuario)",
                user_answer,
            ]
            answer_markdown = "\n\n".join(sections)

            final_ai = AIMessage(
                content=answer_markdown,
                additional_kwargs={"pipeline_internal": True, "node": "summarizer"},
            )

            dev_out = answer_markdown
            deep_out = "\n\n".join([
                "## Resumen deep del pipeline",
                "### ANALYZER",
                analyzer_text,
                "### PLANNER",
                planner_text,
                "### EXECUTOR",
                executor_text,
                "### CATCHER",
                catcher_text,
                "### SUMMARIZER",
                summarizer_text,
                "### RESPUESTA FINAL",
                user_answer,
            ])
            user_out = user_answer

            return {
                "messages": [final_ai],
                "summary": summary_dict,
                "pipeline_summary": summary_dict,
                "dev_out": dev_out,
                "deep_out": deep_out,
                "user_out": user_out,
            }

        tools_summary_text = summarize_tool_runs(user_prompt, runs)

        analyzer = state.get("analyzer") or {}
        subqs = analyzer.get("subqueries") or []
        logic = analyzer.get("propositional_logic") or ""
        input_payload = analyzer.get("input_payload") or {}

        if analyzer:
            analyzer_text_lines = [
                f"Input payload: {input_payload!r}",
                f"Lógica proposicional: {logic or '(no construida)'}",
                f"Subconsultas ({len(subqs)}):",
            ]
            for s in subqs:
                analyzer_text_lines.append(f"- {s}")
            analyzer_text = "\n".join(analyzer_text_lines)
        else:
            analyzer_text = "No se ejecutó ANALYZER o no dejó estado."

        planner_trajs = state.get("planner_trajs", []) or []
        if planner_trajs:
            pl_lines: List[str] = []
            for i, tr in enumerate(planner_trajs, start=1):
                pl_lines.append(f"Subquery {i}: {tr.get('subquery', '')}")
                desc = tr.get("description")
                if desc:
                    pl_lines.append(desc)
            planner_text = "\n".join(pl_lines)
        else:
            planner_text = (
                "No se construyó un plan de herramientas; probablemente se respondió "
                "directamente (o no hubo tool_calls)."
            )

        executor_steps = state.get("executor_steps", []) or []
        if executor_steps:
            ex_lines: List[str] = [
                f"Se ejecutaron {len(executor_steps)} llamadas a herramientas:"
            ]
            for step in executor_steps:
                ex_lines.append(
                    f"- tool_call_id={step['tool_call_id']}, "
                    f"name={step['tool_name']}, args={step['args']!r}"
                )
            executor_text = "\n".join(ex_lines)
        else:
            executor_text = "No se ejecutó ninguna herramienta para esta consulta."

        if runs:
            ca_lines: List[str] = [
                f"Catcher recopiló {len(runs)} resultados de tools."
            ]
            for r in runs:
                ca_lines.append(
                    f"- {r['name']}({r['args']!r}) → output tipo {type(r['output']).__name__}"
                )
            catcher_text = "\n".join(ca_lines)
        else:
            catcher_text = "Catcher no encontró resultados de tools (runs vacío)."

        # Resumen tool-based (para DEV/DEEP)
        summarizer_text = tools_summary_text

        # Respuesta 1:1 en lenguaje natural para USER (tool-first)
        user_answer = build_user_answer(user_prompt, runs)
        if not user_answer:
            # Fallback conservador: si por lo que sea no pudimos mapear nada,
            # devolvemos el resumen de herramientas como antes.
            user_answer = tools_summary_text

        summary_dict: SummaryDict = SummaryDict(
            analyzer=analyzer_text,
            planner=planner_text,
            executor=executor_text,
            catcher=catcher_text,
            summarizer=summarizer_text,
            final_answer=user_answer,
        )

        # Esta respuesta (answer_markdown) es la vista "dev" con todo el pipeline.
        sections = [
            "## Resumen del pipeline",
            "### ANALYZER",
            analyzer_text,
            "### PLANNER",
            planner_text,
            "### EXECUTOR",
            executor_text,
            "### CATCHER",
            catcher_text,
            "### SUMMARIZER (basado en herramientas)",
            summarizer_text,
            "### RESPUESTA FINAL (modo usuario)",
            user_answer,
        ]
        answer_markdown = "\n\n".join(sections)

        final_ai = AIMessage(
            content=answer_markdown,
            additional_kwargs={"pipeline_internal": True, "node": "summarizer"},
        )

        # Además rellenamos dev_out / deep_out / user_out:
        dev_out = answer_markdown
        deep_out = "\n\n".join([
            "## Resumen deep del pipeline",
            "### ANALYZER",
            analyzer_text,
            "### PLANNER",
            planner_text,
            "### EXECUTOR",
            executor_text,
            "### CATCHER",
            catcher_text,
            "### SUMMARIZER",
            summarizer_text,
            "### RESPUESTA FINAL",
            user_answer,
        ])
        user_out = user_answer

        return {
            "messages": [final_ai],
            "summary": summary_dict,
            "pipeline_summary": summary_dict,
            "dev_out": dev_out,
            "deep_out": deep_out,
            "user_out": user_out,
        }

    # VALIDATOR (heurística simple, preparada para LLM en el futuro)
    def validator_node(state: State) -> Dict[str, Any]:
        """
        Pequeño validador que mira:
          - si hubo tools,
          - si el Summarizer dijo "no se ejecutó ninguna herramienta",
          - si el final_answer está vacío,
          - y heurísticas ligeras sobre prompts tabulares/contratos.

        Marca all_covered=False en casos sospechosos.
        Más adelante se puede reemplazar por un LLM que reciba:
          (user_prompt, tool_runs, final_answer) y devuelva ValidatorResult.
        """
        user_prompt = state.get("user_prompt") or ""
        summary = state.get("pipeline_summary") or state.get("summary") or {}
        final_answer = summary.get("final_answer") or ""
        summarizer_text = summary.get("summarizer") or ""
        runs = state.get("tool_runs", []) or []

        # 3) Guardrail en VALIDATOR: auto-reparación (modo sin tools)
        bad_templates = (
            "no se invocó ninguna herramienta",
            "no puedo responder con garantías",
            "sin herramientas no puedo",
        )
        if runs == [] and any(t in final_answer.strip().lower() for t in bad_templates):
            last_ai = find_last_assistant_real(state.get("messages", []) or [])
            raw = state.get("llm_raw_out") or (_coerce_content_str(getattr(last_ai, "content", "")) if last_ai else "")
            direct = state.get("llm_clean_out") or strip_think(raw)

            if direct:
                final_answer = direct
                try:
                    summary["final_answer"] = direct
                except Exception:
                    pass

                # también reparamos user_out si estaba “apagado”
                state["user_out"] = direct

        all_covered = True
        reasons: List[str] = []

        if not final_answer.strip():
            all_covered = False
            reasons.append("La respuesta final está vacía.")

        if "No se invocó ninguna herramienta" in summarizer_text and runs:
            all_covered = False
            reasons.append(
                "Inconsistencia: el SUMMARIZER dice que no hubo tools, "
                "pero tool_runs no está vacío."
            )

        # Heurística: prompt tabular sin tools
        if runs == [] and user_prompt:
            if any(tok in user_prompt.lower() for tok in ["tabla", "table", "fila", "row", "columna", "column"]):
                all_covered = False
                reasons.append(
                    "El usuario menciona estructuras tabulares pero no se invocaron herramientas; "
                    "puede faltar cálculo determinista sobre tablas/diccionarios."
                )

        # Heurística adicional: caso de contratos sin juicio explícito
        if "contrato" in user_prompt.lower() or "contract" in user_prompt.lower():
            has_judge = any(r.get("name") == "judge_row_with_context" for r in runs)
            if not has_judge:
                all_covered = False
                reasons.append(
                    "El usuario menciona contratos pero no se detectó ninguna ejecución "
                    "de `judge_row_with_context`; podría faltar el juicio fila+contexto."
                )

        if not reasons and all_covered:
            reasons.append("No se detectaron problemas obvios de cobertura.")

        validator: ValidatorResult = {
            "all_covered": all_covered,
            "reasoning": "\n".join(reasons),
        }

        # Mensaje para la traza dev
        validator_msg = AIMessage(
            content=(
                "### VALIDATOR\n\n"
                f"- all_covered: {all_covered}\n"
                f"- reasoning:\n{validator['reasoning']}"
            ),
            additional_kwargs={"pipeline_internal": True, "node": "validator"},
        )

        return {
            "validator": validator,
            "messages": [validator_msg],
            "pipeline_summary": summary,
            "summary": summary,
            "user_out": final_answer if isinstance(final_answer, str) and final_answer.strip() else state.get("user_out"),
        }

    # Router
    def route_from_planner(state: State) -> str:
        messages = state["messages"]
        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
        if not ai_msgs:
            return "summarizer"

        last_ai = ai_msgs[-1]
        if extract_tool_calls(last_ai):
            return "executor"
        return "summarizer"

    # Build graph
    builder = StateGraph(State)

    builder.add_node("analyzer", analyzer_node)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("catcher", catcher_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("validator", validator_node)

    builder.add_edge(START, "analyzer")
    builder.add_edge("analyzer", "planner")
    builder.add_conditional_edges(
        "planner",
        route_from_planner,
        ["executor", "summarizer"],
    )
    builder.add_edge("executor", "catcher")
    builder.add_edge("catcher", "summarizer")
    builder.add_edge("summarizer", "validator")
    builder.add_edge("validator", END)

    graph_app = builder.compile()
    return graph_app


# ─────────────────────────────────────────────
# Logic loader (registro de grafos)
# ─────────────────────────────────────────────

@dataclass
class LogicConfig:
    module: str = "agnostic_agent.logic"
    builder_fn: str = "build_graph_agent"


def load_logic(
    planner_llm: Any,
    tools: List[Any],
    planner_config: Optional[PlannerConfig] = None,
    logic_config: Optional[LogicConfig] = None,
) -> Any:
    """
    Carga y ejecuta la función builder que construye el grafo del agente.

    Por defecto usa este mismo módulo:
        agnostic_agent.logic.build_graph_agent
    """
    cfg = logic_config or LogicConfig()

    if cfg.module == "agnostic_agent.logic":
        builder: Callable[..., Any] = globals().get(cfg.builder_fn)  # type: ignore[assignment]
        if builder is None or not callable(builder):
            raise AttributeError(
                f"No se encontró función builder '{cfg.builder_fn}' en agnostic_agent.logic."
            )
        return builder(planner_llm, tools, planner_config)

    import importlib

    try:
        mod = importlib.import_module(cfg.module)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"No se pudo importar el módulo de lógica '{cfg.module}'."
        ) from e

    builder = getattr(mod, cfg.builder_fn, None)
    if builder is None or not callable(builder):
        raise AttributeError(
            f"El módulo '{cfg.module}' no tiene una función callable '{cfg.builder_fn}'."
        )

    return builder(planner_llm, tools, planner_config)


