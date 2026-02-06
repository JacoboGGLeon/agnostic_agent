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

def analyzer_node(state: State) -> Dict[str, Any]:
    """
    ANALYZER (Rule-Based simple).
    Toma el último mensaje del usuario y rellena `analyzer` en el estado
    con un objeto AnalyzerResult (inputs, questions, logic).
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]
    user_text = _coerce_content_str(getattr(last_msg, "content", ""))

    # Lógica "dummy" equivalente al prompt analyzer, pero en código
    # (En un sistema real, aquí llamarías a un LLM pequeño o regular expressions)
    
    # Heurística simple: consideramos todo el texto como una única query
    # y detectamos si pide "traza" o "explicación".
    wants_trace = "traza" in user_text.lower() or "explicame" in user_text.lower()

    # Detectamos idioma simple
    lang = "es" if " el " in user_text or " la " in user_text else "en"

    # Construimos un AnalyzerResult simulado
    analyzer_res: AnalyzerResult = {
        "input_payload": {"text": user_text},
        "propositional_logic": "q1",
        "subqueries": [user_text],
        "subqueries_logic": ["q1"],
        # Campos extra que podríamos querer pasar al planner
        # "wants_tool_trace": wants_trace,
        # "language": lang,
    }

    return {"analyzer": analyzer_res}


def planner_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    PLANNER (LLM + Tools).
    Invoca al Planner LLM para generar tool_calls basándose en:
    - User prompt (último mensaje).
    - Historial de mensajes (si aplica).
    - KBs disponibles + memory_context.
    """
    # Recuperamos configuración inyectada (o defaults)
    planner_cfg = config.get("configurable", {}).get("planner_config")
    if not planner_cfg:
        planner_cfg = PlannerConfig()

    # 1) Construir System Prompt
    sys_msg = build_planner_system_message(planner_cfg)

    # 2) Añadir contexto de memoria / KBs al system prompt o como mensajes extra
    #    (Aquí lo añadimos como texto en un SystemMessage adicional)
    extra_msgs = []
    
    # Memoria
    mem_ctx = state.get("memory_context")
    mem_str = _format_memory_context(mem_ctx)
    if mem_str:
        extra_msgs.append(SystemMessage(content=f"CONTEXTO DE MEMORIA:\n{mem_str}"))

    # KBs disponibles
    kb_names = state.get("kb_names") or []
    if kb_names:
        kb_str = ", ".join(kb_names)
        extra_msgs.append(
            SystemMessage(
                content=(
                    f"Tienes acceso a las siguientes Fuentes de Conocimiento (KBs): {kb_str}.\n"
                    "Si el usuario pregunta sobre datos contenidos en ellas, usa las tools correspondientes."
                )
            )
        )

    # 3) Invocar LLM
    #    Usamos call_planner_with_retry para robustez en tool_calls
    #    Necesitamos la instancia del LLM. Como es un nodo, idealmente
    #    el LLM se pasa en config o se instancia dentro (cacheado).
    #    Aquí instanciamos uno nuevo (ojo con performance) o usamos uno global si existiera.
    #    Lo correcto es `build_planner_llm(planner_cfg)`.
    
    # Bind tools: necesitamos saber qué tools están disponibles.
    # En esta arquitectura, el Planner LLM debe conocer las tools para alucinar los calls.
    # Recogemos TODAS las tools del registry (o filtrar según config).
    from .tools import get_default_tools
    tools = get_default_tools()  # o filtrar por state["kb_names"] si quisiéramos

    llm = build_planner_llm(planner_cfg)
    llm_with_tools = llm.bind_tools(tools)

    # El input principal son los mensajes
    msgs = state["messages"]

    # Llamada
    ai_msg = call_planner_with_retry(
        planner_llm=llm_with_tools,
        system_message=sys_msg,
        user_or_history_messages=msgs,
        planner_config=planner_cfg,
        extra_system_messages=extra_msgs,
    )

    # Detectar si el modelo decidió "hablar" o "usar tools"
    # (Gracias a bind_tools y call_planner_with_retry, el ai_msg traerá tool_calls si las hay)
    
    # Guardamos el mensaje raw
    return {"messages": [ai_msg]}


def executor_node(state: State) -> Dict[str, Any]:
    """
    EXECUTOR (Tool Runner).
    Ejecuta las tools solicitadas en el último AIMessage.
    """
    messages = state["messages"]
    last_msg = messages[-1]
    
    if not isinstance(last_msg, AIMessage):
        return {}

    # Extraer calls
    calls = extract_tool_calls(last_msg)
    if not calls:
        return {}

    from .tools import TOOL_REGISTRY
    
    results = []
    executor_steps: List[ExecutorStep] = []

    for call in calls:
        call_id = call["id"]
        name = call["name"]
        args = call["args"]

        # Ejecución real
        tool_fn = TOOL_REGISTRY.get(name)
        if tool_fn:
            try:
                # LangChain tools esperan invocación vía .invoke o directa si son funciones decoradas
                # @tool decora la función, así que podemos llamarla o usar .invoke(args)
                if hasattr(tool_fn, "invoke"):
                    out = tool_fn.invoke(args)
                else:
                    out = tool_fn(**args)
            except Exception as e:
                out = f"Error executing {name}: {str(e)}"
        else:
            out = f"Error: Tool '{name}' not found."

        # Guardar resultado como ToolMessage (para que LangGraph sepa que se cerró el call)
        # Nota: content debe ser string para ToolMessage estándar, pero LangGraph admite artifacts.
        # Aquí serializamos a string para el historial, pero el "dato rico" va aparte.
        
        # Serialización segura para el historial
        out_str = _fmt_output(name, out)
        
        results.append(
            ToolMessage(
                tool_call_id=call_id,
                content=out_str,
                name=name
            )
        )

        executor_steps.append(
            {
                "tool_call_id": call_id,
                "tool_name": name,
                "args": args
            }
        )

    return {
        "messages": results,
        "executor_steps": executor_steps,
    }


def catcher_node(state: State) -> Dict[str, Any]:
    """
    CATCHER.
    Recoge los outputs de las tools (ToolMessages) y los consolida en `tool_runs`.
    Esto facilita la vida al Summarizer, que no tiene que parsear el historial.
    """
    messages = state["messages"]
    # Buscamos los ToolMessages nuevos (que no hayamos procesado, o procesamos todos)
    # En un grafo cíclico habría que tener cuidado, pero aquí es lineal por turno.
    
    runs: List[Dict[str, Any]] = []
    
    # Estrategia: barrer historial buscando pares (AIMessage con calls) -> (ToolMessages)
    # O simplemente coger los ExecutorSteps y buscar su output.
    
    steps = state.get("executor_steps", [])
    if not steps:
        return {"tool_runs": []}

    # Mapa rápido id -> output
    out_map = {}
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            out_map[m.tool_call_id] = m.content
    
    for step in steps:
        cid = step["tool_call_id"]
        # Recuperamos output del historial (string)
        # (Si quisiéramos el objeto Python real, tendríamos que haberlo pasado por state aparte,
        #  pero LangGraph serializa. Para este demo, usaremos el string o 
        #  implementaríamos un artifact storage si fuera crítico).
        #  *Mejora*: EXECUTOR podría haber escrito en 'tool_runs' directamente con objetos reales.
        
        # Por simplicidad y coherencia con el notebook v17, asumimos que 
        # EXECUTOR ya hizo el trabajo sucio o que aquí reconstruimos.
        # En el notebook v17 original, Catcher normalizaba.
        
        out_val = out_map.get(cid, "No output found")
        
        runs.append({
            "name": step["tool_name"],
            "args": step["args"],
            "output": out_val
        })

    return {"tool_runs": runs}


def summarizer_node(state: State) -> Dict[str, Any]:
    """
    SUMMARIZER.
    Genera las vistas finales (user_out, deep_out, dev_out).
    1) User View: respuesta natural 1:1.
    2) Deep/Dev View: resumen técnico.
    """
    # Inputs
    analyzer = state.get("analyzer") or {}
    tool_runs = state.get("tool_runs") or []
    user_prompt = state.get("user_prompt") or ""
    
    # 1. User Answer (Rule Based / Template)
    # Intentamos construir una respuesta basada en lo que devolvieron las tools
    user_ans = build_user_answer(user_prompt, tool_runs)
    
    if not user_ans:
        # Fallback: Resumen genérico de tools
        user_ans = summarize_tool_runs(user_prompt, tool_runs)

    # 2. Deep/Dev Summaries
    # Generamos strings descriptivos
    planner_summary = f"Planner executed {len(tool_runs)} tools."
    executor_summary = f"Executed: {', '.join(r['name'] for r in tool_runs)}"
    
    # Rellenamos el estado final
    return {
        "user_out": user_ans,
        "deep_out": f"## DEEP SUMMARY\n\n- Analyzer: {analyzer}\n- Tools: {executor_summary}\n- Final: {user_ans}",
        "dev_out": f"## DEV TRACE\n{json.dumps(state.get('executor_steps', []), indent=2)}",
        "summary": {
            "analyzer": str(analyzer),
            "planner": planner_summary,
            "executor": executor_summary,
            "catcher": "Done",
            "summarizer": "Done",
            "final_answer": user_ans,
        }
    }


def validator_node(state: State) -> Dict[str, Any]:
    """
    VALIDATOR (Rule-Based).
    Verifica si la respuesta (user_out) no está vacía y si parece válida.
    """
    ans = state.get("user_out", "")
    all_covered = bool(ans and len(ans.strip()) > 5)
    
    reasons = []
    if not all_covered:
        reasons.append("Answer is too short or empty.")
    else:
        reasons.append("Answer seems complete.")
        
    validator: ValidatorResult = {
        "all_covered": all_covered,
        "reasoning": "\n".join(reasons),
    }
    
    return {"validator": validator}


# ─────────────────────────────────────────────
# Grafo principal (load_logic)
# ─────────────────────────────────────────────

def load_logic() -> StateGraph:
    """
    Construye y devuelve el StateGraph compilable.
    """
    workflow = StateGraph(State)

    # Nodos
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("catcher", catcher_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("validator", validator_node)

    # Aristas (flujo lineal simple v0.2)
    # START -> analyzer -> planner
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "planner")
    
    # Conditional edge después de planner:
    # Si hubo tool_calls -> executor
    # Si no -> summarizer (respuesta directa del LLM)
    def _route_planner(state: State):
        msgs = state["messages"]
        last_msg = msgs[-1]
        if isinstance(last_msg, AIMessage) and extract_tool_calls(last_msg):
            return "executor"
        return "summarizer"

    workflow.add_conditional_edges("planner", _route_planner)

    # Executor -> catcher -> summarizer
    workflow.add_edge("executor", "catcher")
    workflow.add_edge("catcher", "summarizer")
    
    # Summarizer -> validator -> END
    workflow.add_edge("summarizer", "validator")
    workflow.add_edge("validator", END)

    return workflow


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
