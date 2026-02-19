from __future__ import annotations

"""
Lógica principal (grafo LangGraph) del Agnostic Deep Agent.

Sub-grafos actuales:
- ANALYZER  → descompone el prompt (rule-based sencillo por ahora).
- PLANNER   → usa Planner LLM (OpenAI-compatible) para generar tool_calls.
- EXECUTOR  → ejecuta tools reales (LangChain tools).
- CATCHER   → normaliza las salidas de tools a una lista de runs.
- SUMMARIZER→ construye:
    - respuesta final en modo usuario (user_answer),
    - resumen técnico del pipeline (para vistas deep/dev).
- VALIDATOR → revisa si la respuesta parece cubrir todo lo pedido.

Notas:
- Este módulo sigue usando TypedDict; todavía no está cableado
  a los modelos Pydantic de `schemas.py`.
- Ya integra memoria y knowledge_names en el planner, y deja
  dev_out / deep_out / user_out en el estado.
- Está pensado para casos donde el agente cruza:
    * una tabla de atributos (input A, p.ej. filas de contratos),
    * con tablas de contexto (input B, p.ej. parametrías y
      diccionarios de abreviaturas/definiciones),
    * y, opcionalmente, documentos (OCR de contratos) vía tools
      como context_search_in_csv + rerank_docs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
import json
import os
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
    # Removed: selected_skill, active_skills, active_tools_names per strict user requirement


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
    - user_prompt / session_id / knowledge_names / memory_context:
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
    forced_skill: Optional[str] # ✅ Skill forzada desde UI
    user_prompt: Optional[str]
    session_id: Optional[str]
    knowledge_names: List[str]
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


def _extract_xml_tool_calls(ai_msg: AIMessage) -> List[Dict[str, Any]]:
    """
    Fallback robusto para modelos con salida XML (tipo Qwen/Anthropic):
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
    - tool_calls nativos (OpenAI / standard)
    - additional_kwargs["tool_calls"]
    - XML-based calling (<tool_call>{...}</tool_call>)
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

    return _extract_xml_tool_calls(ai_msg)


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

# Regex mejorada: maneja cierre opcional (si el LLM se corta) y case-insensitve
# (?s) = dot matches newline
# <think>.*? = contenido non-greedy
# (?:</think>|$) = termina en cierre o fin de string
_THINK_RE = re.compile(r"(?s)<think>.*?(?:</think>|$)\s*", flags=re.IGNORECASE)

def strip_think(txt: str) -> str:
    """Elimina <think>...</think> (o hasta fin de string) de forma robusta."""
    if not isinstance(txt, str):
        return ""
    # 1. Intentar eliminar bloques completos o truncados
    cleaned = _THINK_RE.sub("", txt).strip()
    
    # 2. Defensa en profundidad: Si limpiamos todo y queda vacío,
    # significa que el modelo solo pensó y no respondió.
    if not cleaned and txt.strip():
        # Retornamos vacío para que el fallback del Summarizer ("¿Qué te gustaría hacer?") actúe.
        return ""
        
    return cleaned

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
    # Avoid repr() for strings to prevent double escaping in UI
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            # For strings, just output the value (maybe wrapped in quotes for clarity if needed, 
            # but simpler is better for UI display)
            parts.append(f"{k}='{v}'")
        else:
             parts.append(f"{k}={repr(v)}")
    return ", ".join(parts)


def _fmt_output(tool_name: str, v: Any) -> str:
    if isinstance(v, bool):
        return "Sí" if v else "No"
    
    # Generic robust formatter for complex objects
    if isinstance(v, (dict, list, tuple, set)):
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
        tool_name = r["name"]
        args = r["args"]
        output = r["output"]
        arg_str = _fmt_args(args)
        
        # ─── SPECIAL FORMATTING FOR KNOWLEDGE SEARCH ───
        if tool_name == "search_knowledge_base" and isinstance(output, list):
            partes.append(f"\n### 🔍 Resultados de búsqueda (`{arg_str}`)")
            if not output:
                partes.append("_(Sin resultados relevantes)_")
            else:
                first = output[0] if output and isinstance(output[0], dict) else {}
                eff_filter = first.get("effective_source_filter")
                filt_origin = first.get("source_filter_origin")
                if filt_origin or eff_filter:
                    partes.append(
                        f"- `source_filter_origin`: **{filt_origin or 'none'}**  \n"
                        f"- `effective_source_filter`: **{eff_filter or 'none'}**"
                    )

                for idx, item in enumerate(output, start=1):
                    if not isinstance(item, dict):
                        continue
                    if item.get("_meta_only"):
                        continue
                        
                    score = item.get("score", 0.0)
                    src_path = item.get("source_path", "unknown")
                    # Extract basename for cleaner display
                    src_name = os.path.basename(src_path) if src_path else "unknown object"
                    
                    content_md = item.get("md", "").strip()
                    # Truncate content for display if excessively long, though usually chunks are manageable
                    # But for "Deep View" user wants to see what the agent saw.
                    
                    # Markdown Card-like format
                    card = (
                        f"**{idx}. {src_name}** (Relevancia: {score:.2f})"
                        + (
                            f" [Árbol {item.get('search_tree')}, L2={item.get('doc_score', 0):.2f}, L1={item.get('chunk_score', 0):.2f}]"
                            if item.get("search_tree")
                            else ""
                        )
                        + "\n"
                        f"> {content_md.replace(chr(10), '  ' + chr(10))}\n" # Indent content
                    )
                    partes.append(card)
            continue

        # ─── DEFAULT FORMATTING ───
        out_str = _fmt_output(tool_name, output)
        
        # Generic formatting: Code block for large outputs if needed
        if len(out_str) > 100 or "\n" in out_str:
             partes.append(f"- `{tool_name}({arg_str})`:\n```json\n{out_str}\n```")
        else:
             partes.append(f"- `{tool_name}({arg_str})` → **{out_str}**")

    return "\n".join(partes)


# build_user_answer REMOVED (Legacy hardcoded function)


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


def _format_knowledge_hint(knowledge_names: List[str]) -> str:
    if not knowledge_names:
        return ""
    return (
        "KBs disponibles para esta sesión:\n"
        + "\n".join(f"- {name}" for name in knowledge_names)
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
    skill_registry: Any | None = None,  # ✅ Recibimos el registro
):
    """
    Grafo:

        START → ANALYZER → PLANNER
                      ├─(tool_calls)→ EXECUTOR → CATCHER → SUMMARIZER → VALIDATOR → END
                      └─────────────→ SUMMARIZER → VALIDATOR → END
    """
    cfg = planner_config or PlannerConfig()
    base_system_msg = build_planner_system_message(cfg)

    # ANALYZER (LLM-based with Strict JSON)
    def analyzer_node(state: State) -> Dict[str, Any]:
        """
        ANALYZER: Descompone la query y selecciona Skills usando JSON estricto.
        """
        messages = state.get("messages", [])
        user_messages = [m for m in messages if isinstance(m, HumanMessage)]
        last_user = user_messages[-1] if user_messages else None
        user_text = last_user.content if isinstance(last_user, HumanMessage) else ""

        # 📥 INPUTS:
        # user_prompt: Tu pregunta original (el texto que escribes en el chat).
        
        user_prompt = state.get("user_prompt") or user_text

        # INPUT: active_tools (simulamos recepción para cumplir contrato)
        # En esta arquitectura, las tools están en el scope global 'tools' inyectado al builder
        # pero para ser explícitos en el input, las recuperamos del contexto si es posible
        # o simplemente usamos la lista global disponible en el closure.
        active_tools_input = tools # Global scope from closure
        
        # Knowledge bases disponibles
        knowledge_names = state.get("knowledge_names", [])
        knowledge_available = bool(knowledge_names)
        
        # 1. Preparar prompts
        from agnostic_agent.prompts import ANALYZER_SYSTEM_PROMPT, LOGIC_DEFINITIONS
        
        # Inyectar variables en el prompt
        # Nota: available_skills podríamos inyectarlo también si el prompt lo pidiera,
        # pero el nuevo prompt simplificado confía en que el modelo 'sabe' o se le pasa en contexto.
        # Ajustemos para pasarle las skills disponibles si el prompt lo requiere implicitamente
        # o agreguémoslo al user message.
        
        # Para ser robustos, listamos las skills y las pegamos en el prompt si hay placeholder,
        # o simplemente las agregamos al final del system prompt.
        available_skills_txt = ""
        if skill_registry:
            s_list = [f"- {s.name}: {s.description}" for s in skill_registry.list_skills()]
            available_skills_txt = "\n".join(s_list)
        
        # Renderizar prompt
        # El prompt nuevo tiene {user_prompt}, {knowledge_available}, {knowledge_names} y {LOGIC_DEFINITIONS}
        sys_content = ANALYZER_SYSTEM_PROMPT.replace("{user_prompt}", user_prompt) \
                                          .replace("{knowledge_available}", str(knowledge_available)) \
                                          .replace("{knowledge_names}", str(knowledge_names)) \
                                          .replace("{LOGIC_DEFINITIONS}", LOGIC_DEFINITIONS) \
                                          .replace("{AVAILABLE_SKILLS}", available_skills_txt or "[]")
        
        if available_skills_txt:
            sys_content += f"\n\nSKILLS DISPONIBLES:\n{available_skills_txt}"
            
        # Refuerzo para latencia: si el usuario desactivó el pensamiento
        if cfg and not cfg.enable_thinking:
            sys_content += "\n\nCRITICAL: DO NOT use <think> tags. Respond ONLY with the JSON block."
            
        sys_msg = SystemMessage(content=sys_content)
        # Enviamos un mensaje dummy de usuario para activar la generación
        user_msg = HumanMessage(content="Analiza mi petición y genera el JSON.")

        # 2. Invocar LLM (o Bypass si hay Forced Skill)
        selected_skills = []
        subqueries = [user_prompt]
        logic_form = "q1"
        
        # --- FORCED SKILL LOGIC ---
        forced_skill = state.get("forced_skill")
        if forced_skill and forced_skill != "Auto (Analyzer)":
             print(f"[ANALYZER] 🔒 Forced Skill active: {forced_skill}")
             
             # Bypass LLM invocation completely for decomposition.
             # We assume the user's prompt is the single subquery to be solved by this skill.
             selected_skills = [forced_skill]
             subqueries = [user_prompt]
             logic_form = "q1"
             
             # Early return structure (construction similar to end of function)
             subqueries_logic = ["q1"]
             analyzer = {
                "input_payload": {"user_prompt": user_prompt},
                "propositional_logic": logic_form,
                "subqueries": subqueries,
                "subqueries_logic": subqueries_logic,
             }
             analyzer_msg = AIMessage(
                content=f"### ANALYZER (Forced Mode)\nSkills: {selected_skills}\nSubqueries: {subqueries}",
                additional_kwargs={"pipeline_internal": True, "node": "analyzer"}
             )
             return {
                "analyzer": analyzer,
                "_active_skills_internal": selected_skills,
                "messages": [analyzer_msg]
             }
        # --------------------------

        try:
            response = planner_llm.invoke([sys_msg, user_msg])
            content = response.content
            
            # 3. Parseo Robusto de JSON
            # Limpiar bloques markdown ```json ... ```
            if "```" in content:
                import re
                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```\s*", "", content)
            
            # Additional cleanup for Qwen sometimes returning descriptive text before JSON
            content = content.strip()
            if not content.startswith("{") and "{" in content:
                 content = content[content.find("{"):]
                 if "}" in content:
                      content = content[:content.rfind("}")+1]

            data = json.loads(content)
            
            subqueries = data.get("subqueries", [user_prompt])
            logic_form = data.get("logic_form", "q1")
            selected_skills = data.get("selected_skills", [])
            
            # --- FALLBACK AGNOSTICO ---
            # Si hay KBs disponibles y el modelo no seleccionó ninguna skill (o lista vacía),
            # FORZAMOS 'semantic_researcher' para evitar uso de tools alucinadas (Skills-First).
            if not selected_skills and knowledge_available:
                print("[ANALYZER] ℹ️ Model returned no skills but Knowledge Bases are available.")
                if skill_registry and skill_registry.get_skill("semantic_researcher"):
                     print("[ANALYZER] 🛡️ Enforcing 'semantic_researcher' to restrict tool usage.")
                     selected_skills = ["semantic_researcher"]
                else:
                     print("[ANALYZER] ⚠️ 'semantic_researcher' not found in registry. Running in generalist mode.")
            # ------------------------
            
            print(f"[ANALYZER] JSON OK. Skills: {selected_skills}")
            
        except Exception as e:
            print(f"[ANALYZER] Error parsing JSON: {e}. Content: {getattr(response, 'content', '')[:100]}...")
            # Fallback simple: lista vacía
            if knowledge_available:
                 print(f"[ANALYZER] Error fallback: leaving skills empty.")
                 selected_skills = []
        
        # 4. Construir resultado
        subqueries_logic = [f"q{i+1}" for i in range(len(subqueries))]
        
        # 📤 OUTPUTS:
        # subqueries: La pregunta descompuesta en pasos lógicos.
        # propositional_logic: La relación entre las subconsultas.
        
        analyzer: AnalyzerResult = {
            "input_payload": {"user_prompt": user_prompt},
            "propositional_logic": logic_form,
            "subqueries": subqueries,
            "subqueries_logic": subqueries_logic,
        }
        
        analyzer_msg = AIMessage(
            content=f"### ANALYZER (JSON Mode)\nSkills: {selected_skills}\nSubqueries: {subqueries}",
            additional_kwargs={"pipeline_internal": True, "node": "analyzer"}
        )
        
        # INTERNAL STATE: Skills activas para que el Planner filtre tools
        # (No forma parte del contrato oficial AnalyzerResult, pero es necesario para el filtrado)
        return {
            "analyzer": analyzer,
            "_active_skills_internal": selected_skills,
            "messages": [analyzer_msg]
        }

    def _format_rich_context(skills_reg, tools_list, knowledge_list, exclude_skills=None) -> str:
        """
        Construye el Contexto Estructurado (Rich Registry) con metadata/esquemas detallados.
        
        Compliance con Requerimientos:
        - Skills: Lista de skills activas y expansion a ellas (cómo usar tools sobre knowledge).
        - Tools: Definiciones (JSON schema) y docstrings (input, descripcion, output).
        - Knowledge: Descripciones literales de la database.
        """
        lines = ["== CONTEXTO DEL SISTEMA (Capabilities) ==", ""]

        # 1. SKILLS (Expansión y Guía)
        lines.append("### 🧩 SKILLS (Estrategias Activas)")
        if skills_reg:
            all_skills = skills_reg.list_skills()
            if all_skills:
                for s in all_skills:
                    if exclude_skills and s.name in exclude_skills:
                        continue
                    
                    # Expansión: Cómo usar tools sobre knowledges
                    knowledge_hint = ""
                    if s.knowledge:
                        knowledge_hint = f" -> Opera sobre Knowledge: {s.knowledge}"
                    
                    tools_hint = ""
                    if s.tools:
                        tools_hint = f" -> Orquesta Tools: {s.tools}"

                    lines.append(
                        f"@skill {{name={s.name}}}"
                    )
                    lines.append(f"  Description: {s.description}")
                    lines.append(f"  Expansion: {tools_hint}{knowledge_hint}")
                    lines.append("")
            else:
                lines.append("(No skills loaded)")
        else:
            lines.append("(Skill Registry not available)")
        lines.append("")

        # 2. TOOLS (Schema + Docstring completo)
        lines.append("### 🛠 TOOLS (Funciones ejecutables)")
        if tools_list:
            for t in tools_list:
                name = getattr(t, "name", "tool")
                desc = getattr(t, "description", str(t)) # Docstring principal / Descripción
                
                # Metadata extendida
                metadata = getattr(t.func if hasattr(t, 'func') else t, '_agnostic_metadata', None)
                
                if metadata:
                    mode = metadata.get('mode', 'public')
                    input_schema = metadata.get('input_schema')
                    output_schema = metadata.get('output_schema', {})
                else:
                    mode = 'public'
                    output_schema = {}
                    input_schema = None
                    if hasattr(t, "args_schema") and t.args_schema:
                        try:
                            input_schema = t.args_schema.schema_json()
                        except Exception:
                            input_schema = str(t.args_schema)
                
                input_str = json.dumps(input_schema) if input_schema else "Any"
                output_str = json.dumps(output_schema) if output_schema else "{}"
                
                lines.append(f"@tool {{name={name}, mode={mode}}}")
                lines.append(f"  Description/WhatItDoes: {desc}")
                lines.append(f"  Input Schema: {input_str}")
                lines.append(f"  Output Schema: {output_str}")
                lines.append("")
        else:
            lines.append("(No tools available)")
        lines.append("")

        # 3. KNOWLEDGE (Descripciones Literales)
        lines.append("### 📚 KNOWLEDGE (Bases de Datos)")
        if knowledge_list:
            for knowledge in knowledge_list:
                # knowledge es un dict que viene de la base de datos (agent.py)
                knowledge_name = knowledge.get("name", "unknown")
                knowledge_desc = knowledge.get("description", "Sin descripción")
                
                lines.append(f"@knowledge {{name={knowledge_name}}}")
                lines.append(f"  Description: {knowledge_desc}") # Literal de la DB
                lines.append("")
        else:
            lines.append("(No knowledge bases active)")
        
        return "\n".join(lines)

    # PLANNER v2 (DAG JSON)
    def planner_node(state: State) -> Dict[str, Any]:
        """
        PLANNER: Genera un plan DAG en JSON y lo convierte a tool_calls lineales.
        """
        msgs: List[AnyMessage] = state["messages"]
        
        # Contextos
        knowledge_names = state.get("knowledge_names") or []
        # RECUPERAR OBJETOS KNOWLEDGE COMPLETOS (para tener description)
        # agent.py inyecta "knowledge_selected" como lista de dicts
        knowledge_selected = state.get("knowledge_selected") or []
        
        # INPUT REFINEMENT: Planner solo debe depender de subqueries y rich_context
        
        analyzer = state.get("analyzer") or {}
        # 1. INPUT: subqueries (Extraído explícitamente del output del Analyzer)
        subqs = analyzer.get("subqueries") or []
        
        # 1.1 FILTRADO ESTRICTO DE TOOLS POR SKILLS
        # Las skills activas están en _active_skills_internal (no en AnalyzerResult oficial)
        active_skills = state.get("_active_skills_internal") or []
        skill_mode = len(active_skills) > 0
        
        # Si hay skills activas, SOLO mostramos las tools que ellas declaran
        required_tool_names = set()
        required_knowledge_names = set()
        
        if skill_mode and skill_registry:
            for skill_name in active_skills:
                skill = skill_registry.get_skill(skill_name)
                if skill:
                    if skill.tools:
                        required_tool_names.update(skill.tools)
                    if skill.knowledge:
                        required_knowledge_names.update(skill.knowledge)
        
        # Filtrado estricto: Si hay skills, SOLO las tools de esas skills
        if skill_mode and required_tool_names:
            active_tools = [t for t in tools if t.name in required_tool_names]
        else:
            # Sin skills: modo genérico, todas las tools disponibles
            active_tools = tools
        
        # Filtrar Knowledge activas
        if skill_mode and required_knowledge_names and "*" not in required_knowledge_names:
            active_knowledge_objects = [knowledge for knowledge in knowledge_selected if knowledge.get("name") in required_knowledge_names]
        else:
            active_knowledge_objects = knowledge_selected

        # 2. Construir Contexto
        # Pasamos active_skills para excluirlas del contexto (evitar recursión)
        rich_context_text = _format_rich_context(
            skill_registry, 
            active_tools, 
            active_knowledge_objects,
            exclude_skills=active_skills if skill_mode else None
        )
        
        # 3. Preparar Prompt DAG
        from agnostic_agent.prompts import PLANNER_DAG_SYSTEM_PROMPT
        
        # Inyectar variables
        sys_content = PLANNER_DAG_SYSTEM_PROMPT
        
        # Si el usuario desactivó el pensamiento, forzamos al modelo en el prompt
        if cfg and not cfg.enable_thinking:
            sys_content += "\n\nCRITICAL: DO NOT use <think> tags. Respond ONLY with the JSON DAG block."
        
        # Crear mensaje de usuario con los inputs estructurados
        user_msg_content = f"""CONTEXTO DISPONIBLE:
{rich_context_text}

TAREA:
Genera el DAG para resolver: {json.dumps(subqs, ensure_ascii=False)}"""

        user_msg = HumanMessage(content=user_msg_content)
        sys_msg = SystemMessage(content=sys_content)
        

        # 4. Invocar Planner LLM (Pasando historia relevante)
        # Filtramos solo mensajes reales (Human/AI) para no saturar con mensajes internos del pipeline
        history = [m for m in msgs if not _is_pipeline_internal_ai(m)]
        
        tool_calls = []
        llm_raw_out = ""
        

        # 3. Iterar Subqueries (Query-by-Query Planning)
        # CANONICAL IMPLEMENTATION: Loop over each subquery to generate specific plans
        from agnostic_agent.prompts import PLANNER_DAG_SYSTEM_PROMPT
        
        # Inyectar variables globales
        sys_content = PLANNER_DAG_SYSTEM_PROMPT
        
        # Si el usuario desactivó el pensamiento, forzamos al modelo en el prompt
        if cfg and not cfg.enable_thinking:
            sys_content += "\n\nCRITICAL: DO NOT use <think> tags. Respond ONLY with the JSON DAG block."
        
        # Preparar LLM (Re-bind si es necesario)
        current_llm = planner_llm
        if skill_mode and active_tools:
            # Desempaquetar RunnableBinding si existe
            base_model = getattr(planner_llm, "bound", planner_llm)
            current_llm = base_model.bind_tools(active_tools)
            print(f"[PLANNER] 🔒 Skill Mode Active. Re-bound LLM to {len(active_tools)} tools.")
        
        history = [m for m in msgs if not _is_pipeline_internal_ai(m)]
        
        all_tool_calls: List[Dict[str, Any]] = []
        plan_trajs: List[PlannerTrajectory] = []
        
        global_llm_clean: str = ""
        global_llm_raw: str = ""
        
        # Fallback si no hay subqueries (usar prompt directo)
        if not subqs:
             user_messages = [m for m in msgs if isinstance(m, HumanMessage)]
             last_user = user_messages[-1] if user_messages else None
             if isinstance(last_user, HumanMessage):
                 subqs = [last_user.content]

        # LOOP PRINCIPAL: Query-by-Query
        seen_calls_keys = set()
        
        for i, subq in enumerate(subqs, start=1):
            try:
                print(f"[PLANNER] Planning for Subquery {i}/{len(subqs)}: {subq}")
                
                # Construir prompt específico
                user_msg_content = f"""CONTEXTO DISPONIBLE:
{rich_context_text}

TAREA ACTUAL (Subquery {i} de {len(subqs)}):
Genera el DAG exclusivo para resolver: "{subq}"
"""
                user_msg = HumanMessage(content=user_msg_content)
                
                # Enviar [sys, ...history, user]
                response = current_llm.invoke([SystemMessage(content=sys_content)] + history[:-1] + [user_msg])
                
                current_raw = response.content
                global_llm_raw += f"\n\n--- Subquery {i}: {subq} ---\n{current_raw}"
                
                # Parseo DAG
                content_cleaned = strip_think(current_raw).strip()
                global_llm_clean += f"\n\n--- Plan {i}: {subq} ---\n{content_cleaned}"
                
                # Limpieza JSON
                if "```" in content_cleaned:
                    import re
                    content_cleaned = re.sub(r"```json\s*", "", content_cleaned)
                    content_cleaned = re.sub(r"```\s*", "", content_cleaned)
                
                start_brace = content_cleaned.find("{")
                end_brace = content_cleaned.rfind("}")
                
                dag_steps = []
                if start_brace != -1 and end_brace != -1:
                    json_str = content_cleaned[start_brace : end_brace + 1]
                    try:
                        dag_data = json.loads(json_str)
                        dag_steps = dag_data.get("dag", [])
                    except json.JSONDecodeError:
                        print(f"[PLANNER] ⚠️ JSON Decode Error for subquery {i}")
                
                # Procesar pasos del DAG
                allowed_tool_names = {t.name for t in active_tools} if active_tools else None
                subq_calls = []
                
                for step in dag_steps:
                    t_name = step.get("tool")
                    t_args = step.get("args", {})
                    # RE-GENERATE ID to ensure uniqueness across consolidated subqueries
                    # The LLM often restarts at "step_1" for each subquery.
                    original_id = step.get("step_id") or "step_?"
                    t_id = f"{original_id}_{i}_{str(uuid.uuid4())[:4]}"
                    
                    if t_name:
                        # STRICT SKILL CHECK
                        if skill_mode and allowed_tool_names and t_name not in allowed_tool_names:
                            print(f"[PLANNER] ⛔ Tool '{t_name}' BLOCKED (Not in skill).")
                            continue
                        
                        # DEDUPLICATION CHECK
                        # Usamos nombre + args como clave única
                        args_sorted = json.dumps(t_args, sort_keys=True)
                        dedup_key = (t_name, args_sorted)
                        
                        if dedup_key in seen_calls_keys:
                            print(f"[PLANNER] ⚠️ Duplicate call skipped: {t_name}")
                            continue
                        seen_calls_keys.add(dedup_key)
                        
                        call_obj = {
                            "name": t_name,
                            "args": t_args,
                            "id": t_id,
                            "type": "tool_call"
                        }
                        subq_calls.append(call_obj)
                        all_tool_calls.append(call_obj)
                
                # Generar descripción para PlannerTrajectory
                desc_lines = []
                if not subq_calls:
                    desc_lines.append("No tools needed or planning specific to this query.")
                else:
                     for tc in subq_calls:
                         desc_lines.append(f"Call `{tc['name']}`")
                
                plan_trajs.append(PlannerTrajectory(
                    subquery=subq,
                    description="\n".join(desc_lines)
                ))
                
            except Exception as e:
                print(f"[PLANNER] ❌ Error planning subquery {i}: {e}")
                plan_trajs.append(PlannerTrajectory(subquery=subq, description=f"Error: {e}"))

        # 7. Construir AIMessage final consolidado
        print(f"[PLANNER] Total consolidated tool calls: {len(all_tool_calls)}")
        
        ai_msg = AIMessage(
            content=global_llm_clean.strip(),
            tool_calls=all_tool_calls,
            additional_kwargs={"dag_raw": global_llm_raw}
        )

        return {
            "messages": [ai_msg],
            "planner_trajs": plan_trajs,
            "llm_raw_out": global_llm_raw,
            "llm_clean_out": global_llm_clean,
        }

    # EXECUTOR HELPERS
    def _resolve_dependency_arg(val: Any, results: Dict[str, Any]) -> Any:
        """
        Resuelve referencias tipo '$step_1.output' usando el diccionario de resultados previos.
        Soporta anidación en listas y dicts.
        """
        if isinstance(val, str) and val.strip().startswith("$"):
            ref = val.strip()[1:]  # quitar $
            parts = ref.split(".")
            step_id = parts[0]
            
            if step_id in results:
                res = results[step_id]
                # Si piden un campo específico (ej: $step_1.output.id)
                if len(parts) > 1:
                    field = parts[1]
                    # 'output' es la keyword estándar para el resultado completo, 
                    # pero si el resultado es un dict, permitimos acceso a subcampos
                    if field == "output":
                        return res
                    if isinstance(res, dict):
                        return res.get(field, val) # Fallback al literal si no existe
                return res
            # Si no encontramos el step_id, devolvemos el literal (posible fallo posterior)
            return val
            
        if isinstance(val, list):
            return [_resolve_dependency_arg(v, results) for v in val]
        
        if isinstance(val, dict):
            return {k: _resolve_dependency_arg(v, results) for k, v in val.items()}
            
        return val

    def _repair_tool_args(tool_obj: Any, raw_args: Any) -> Any:
        """
        Normaliza args al esquema real de la tool para tolerar planes con claves genéricas
        (por ejemplo: {"arg_name": "..."} para tools de un solo parámetro).
        """
        if not isinstance(raw_args, dict):
            return raw_args

        args = dict(raw_args)
        schema = getattr(tool_obj, "args", None)
        if not isinstance(schema, dict):
            return args

        expected_fields = list(schema.keys())
        if not expected_fields:
            return args

        # Caso típico: planner devuelve {"arg_name": ...} y la tool espera un único campo.
        if "arg_name" in args and "arg_name" not in expected_fields and len(expected_fields) == 1:
            target = expected_fields[0]
            args[target] = args.get("arg_name")
            args.pop("arg_name", None)
            return args

        # Si el plan trae un único campo no esperado y la tool espera solo uno, remap defensivo.
        if len(expected_fields) == 1:
            target = expected_fields[0]
            if target not in args and len(args) == 1:
                only_key = next(iter(args.keys()))
                if only_key not in expected_fields:
                    args[target] = args.pop(only_key)

        return args


    # EXECUTOR
    def executor_node(state: State) -> Dict[str, Any]:
        messages = state["messages"]

        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
        if not ai_msgs:
            print("[EXECUTOR] No AIMessage found in state.")
            return {"messages": [], "executor_steps": []}

        ai_plan = ai_msgs[-1]
        
        # 1. Intentar sacar tool_calls explícitos (LangChain attr)
        tool_calls = getattr(ai_plan, "tool_calls", None)
        
        # 2. Fallback a extractor manual
        if not tool_calls:
             tool_calls = extract_tool_calls(ai_plan)
             
        if not tool_calls:
            print("[EXECUTOR] No tool calls found involved in message. Skipping.")
            return {"messages": [], "executor_steps": []}

        print(f"[EXECUTOR] Executing {len(tool_calls)} tools...")
        
        tool_msgs: List[ToolMessage] = []
        exec_steps: List[Dict[str, Any]] = []
        
        # Diccionario local de resultados para resolución de dependencias
        # step_id -> result
        local_results: Dict[str, Any] = {}

        for tc in tool_calls:
            # Soportar dict o objeto ToolCall
            if isinstance(tc, dict):
                name = tc.get("name")
                args_raw = tc.get("args", {}) or {}
                t_id = tc.get("id")
            else:
                name = getattr(tc, "name", "")
                args_raw = getattr(tc, "args", {}) or {}
                t_id = getattr(tc, "id", "")
            
            # ─────────────────────────────────────────────
            # RESOLUCIÓN DE VARIABLES (Ambient Context)
            # ─────────────────────────────────────────────
            args = _resolve_dependency_arg(args_raw, local_results)
            
            print(f"[EXECUTOR] Running tool: {name} with resolved args: {args}")

            try:
                tool_obj = next(t for t in tools if t.name == name)
                args = _repair_tool_args(tool_obj, args)
                observation = tool_obj.invoke(args)
            except StopIteration:
                observation = {"error": f"Tool '{name}' no encontrada."}
            except Exception as e:
                observation = {"error": f"Excepción ejecutando tool '{name}': {e!r}"}

            # Guardar resultado para pasos posteriores
            if t_id:
                local_results[t_id] = observation

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

            tool_msgs.append(
                ToolMessage(
                    content=payload,
                    tool_call_id=t_id,
                    name=name
                )
            )
            
            exec_steps.append({
                "tool_name": name,
                "args": args,
                "tool_call_id": t_id
            })

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

        # Extraer analyzer al inicio para tener scope en todo el nodo
        analyzer = state.get("analyzer") or {}

        # 2) Parche SUMMARIZER (regla de oro):
        # Si NO hay tools (runs vacío) y el último AI NO tiene tool_calls,
        # user_out debe ser la salida directa del LLM (limpia de <think>).
        if not runs:
            # (Código modo sin tools, se mantiene igual)
            last_ai = find_last_assistant_real(messages)
            last_ai_has_tools = bool(extract_tool_calls(last_ai)) if last_ai else False

            llm_raw = state.get("llm_raw_out") or (_coerce_content_str(getattr(last_ai, "content", "")) if last_ai else "")
            llm_clean = state.get("llm_clean_out") or strip_think(llm_raw)

            # DETECTAR SI ES UN JSON DAG VACÍO (Artifact of Planner Node)
            is_empty_dag = False
            # Check for specific artifacts of planner_node output structure
            # It usually looks like: "\n\n--- Plan 1: ... ---\n{\n "dag": []\n}"
            if '"dag": []' in llm_clean or "'dag': []" in llm_clean:
                 is_empty_dag = True
            
            if is_empty_dag:
                 # FALLBACK CONVERSACIONAL
                 # El planner dijo "no necesito tools". Ahora responde al usuario.
                 fallback_sys = (
                     "Eres un asistente servicial y agnóstico.\n"
                     "El usuario te ha dicho algo que NO requiere herramientas externas.\n"
                     "Responde de forma natural, útil y amable en el idioma del usuario.\n"
                     "NO inventes información."
                 )
                 
                 try:
                     # Usamos el base_model (sin tools bound) para evitar loops o tool execution
                     # planner_llm está en el closure de build_graph_agent
                     base_chat_model = getattr(planner_llm, "bound", planner_llm)
                     
                     fallback_reply = base_chat_model.invoke([
                        SystemMessage(content=fallback_sys),
                        HumanMessage(content=user_prompt)
                     ])
                     fallback_content = getattr(fallback_reply, "content", "")
                     # Strip thinking locally just in case
                     user_answer = strip_think(fallback_content)
                 except Exception as e:
                     user_answer = f"Hola. He recibido tu mensaje: '{user_prompt}'"

            elif last_ai_has_tools:
                user_answer = (
                    "Se planificaron llamadas a herramientas, pero no se obtuvo ninguna salida. "
                    "Revisa EXECUTOR/CATCHER o el registro de tools."
                )
            else:
                # Mejor UX: Si el modelo pensó pero no respondió (todo era <think>), avisar.
                if not llm_clean and llm_raw and llm_raw.strip():
                    user_answer = (
                        "_(El modelo generó un razonamiento interno pero no una respuesta final. "
                        "Ver pestaña 'Thinking' en el Inspector)_"
                    )
                else:
                    user_answer = llm_clean or "¿Qué te gustaría hacer?"

            tools_summary_text = summarize_tool_runs(user_prompt, runs)
            
            # --- Reconstrucción de metadatos (para simplificar, reusemos lógica) ---
            # analyzer ya está definido arriba
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
            summarizer_text = "No se invocaron herramientas. Respuesta directa del modelo (passthrough)."

            summary_dict: SummaryDict = SummaryDict(
                analyzer=analyzer_text,
                planner=planner_text,
                executor=executor_text,
                catcher=catcher_text,
                summarizer=summarizer_text,
                final_answer=user_answer,
            )

        else:
            # SÍ HAY TOOLS (runs > 0)
            tools_summary_text = summarize_tool_runs(user_prompt, runs)
            
            # --- HYBRID/PROACTIVE MODE (Always) ---
            # Ya no chequeamos cfg.policy_mode == "hybrid" porque es el único modo.
            
            # Sintetizar con LLM (usando planner_llm)
            
            # Recuperar instrucciones de skills activas para el Summarizer
            skill_gen_instructions = ""
            if analyzer and analyzer.get("active_skills"):
                if skill_registry:
                    for s_name in analyzer["active_skills"]:
                        skill = skill_registry.get_skill(s_name)
                        if skill:
                            skill_gen_instructions += f"\n\n--- INSTRUCCIONES ESPECÍFICAS ({s_name}) ---\n{skill.instructions}"

            hybrid_sys = (
                "Eres un asistente que responde preguntas basándose ESTRICTAMENTE en la información provista "
                "por las herramientas (Contexto).\n"
                "Tu objetivo es transformar los datos crudos de las herramientas en una respuesta natural, "
                "fluida y útil para el usuario.\n"
                "- NO agregues información externa que no esté en el contexto.\n"
                "- SI el contexto está vacío o no es relevante, indícalo.\n"
                "- Citas: Si es posible, menciona la fuente (ej: 'según el documento X...').\n"
                "- Responde en el mismo idioma del usuario."
            )
            
            if skill_gen_instructions:
                hybrid_sys += skill_gen_instructions

            
            # Refuerzo para latencia
            if cfg and not cfg.enable_thinking:
                hybrid_sys += "\n\nCRITICAL: DO NOT use <think> tags. Respond ONLY with the final natural language answer."
            
            hybrid_user_msg = (
                f"Pregunta del usuario: {user_prompt}\n\n"
                f"Información de Herramientas (Contexto):\n{tools_summary_text}\n\n"
                "Respuesta:"
            )
            try:
                # Usamos planner_llm para sintetizar
                hrm = planner_llm.invoke([
                    SystemMessage(content=hybrid_sys), 
                    HumanMessage(content=hybrid_user_msg)
                ])
                user_answer = hrm.content
                
                # ═══════════════════════════════════════════════════════════
                # CAPTURAR REASONING del FINAL ANSWER (agnóstico)
                # ═══════════════════════════════════════════════════════════
                
                # 1. Intentar extrar <think> del contenido (Texto crudo)
                import re
                think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
                match = think_pattern.search(user_answer)
                
                reasoning_from_final = ""
                
                if match:
                    reasoning_from_final = match.group(1).strip()
                    # Limpiamos el <think> del user_answer para que no salga en el chat limpio
                    user_answer = think_pattern.sub("", user_answer).strip()
                
                # 2. Si no hay <think> en texto, buscar en metadatos del proveedor (DeepSeek/Azure)
                if not reasoning_from_final:
                    if hasattr(hrm, 'additional_kwargs') and isinstance(hrm.additional_kwargs, dict):
                        reasoning_from_final = (
                            hrm.additional_kwargs.get("reasoning_content") or 
                            hrm.additional_kwargs.get("reasoning") or 
                            hrm.additional_kwargs.get("thoughts") or 
                            ""
                        )
                
                # Si hay reasoning, crear un AIMessage dedicado para el Inspector
                if reasoning_from_final and isinstance(reasoning_from_final, str) and reasoning_from_final.strip():
                    # Mensaje con el reasoning para que el Inspector lo muestre
                    thinking_msg = AIMessage(
                        content="",  # Contenido vacío, solo queremos el reasoning
                        additional_kwargs={
                            "reasoning_content": reasoning_from_final.strip(),
                            "final_answer_thinking": True,  # Marca para identificarlo
                        }
                    )
                    # Agregar al state para que el Inspector lo capture
                    state.setdefault("messages", []).append(thinking_msg)
                    
            except Exception as e:
                user_answer = f"(Error en síntesis híbrida: {e})\n\nResumen crudo:\n{tools_summary_text}"

            # --- Reconstrucción de metadatos (Analyzer, Planner, Executor, Catcher) ---
            analyzer = state.get("analyzer") or {}
            subqs = analyzer.get("subqueries") or []
            logic = analyzer.get("propositional_logic") or ""
            input_payload = analyzer.get("input_payload") or {}
            
            # (Copia de lógica de metadatos para consistencia)
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
                planner_text = "No se construyó un plan de herramientas."

            executor_steps = state.get("executor_steps", []) or []
            if executor_steps:
                ex_lines_list: List[str] = [
                    f"Se ejecutaron {len(executor_steps)} llamadas a herramientas:"
                ]
                for step in executor_steps:
                    ex_lines_list.append(
                        f"- tool_call_id={step['tool_call_id']}, "
                        f"name={step['tool_name']}, args={step['args']!r}"
                    )
                executor_text = "\n".join(ex_lines_list)
            else:
                executor_text = "No se ejecutó ninguna herramienta para esta consulta."

            if runs:
                ca_lines_list: List[str] = [
                    f"Catcher recopiló {len(runs)} resultados de tools."
                ]
                for r in runs:
                    ca_lines_list.append(
                        f"- {r['name']}({r['args']!r}) → output tipo {type(r['output']).__name__}"
                    )
                catcher_text = "\n".join(ca_lines_list)
            else:
                catcher_text = "Catcher no encontró resultados de tools (runs vacío)."

            summarizer_text = tools_summary_text

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
        # ═══════════════════════════════════════════════════════════════════
        # AGNOSTIC FIX: Strip <think> tags from user_out
        # ═══════════════════════════════════════════════════════════════════
        # El user_out debe ser limpio (sin <think>) para ser agnóstico:
        # - Modelos con reasoning (Qwen3, DeepSeek) → strip <think>
        # - Modelos sin reasoning (GPT-4, etc.) → no afecta
        user_out = strip_think(user_answer)

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

    # Router (Updated Debug)
    def route_from_planner(state: State) -> str:
        messages = state["messages"]
        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
        if not ai_msgs:
            return "summarizer"

        last_ai = ai_msgs[-1]
        
        # Check explicit tool_calls first (LangChain standard)
        tc = getattr(last_ai, "tool_calls", None)
        if tc and isinstance(tc, list) and len(tc) > 0:
            print(f"[ROUTER] Going to EXECUTOR. Found {len(tc)} tool_calls.")
            return "executor"
            
        # Fallback to current extractor
        extracted = extract_tool_calls(last_ai)
        if extracted:
            print(f"[ROUTER] Going to EXECUTOR. Extracted {len(extracted)} tool_calls.")
            return "executor"
            
        print("[ROUTER] No tool calls found. Going to SUMMARIZER.")
        print(f"[ROUTER DEBUG] Content start: {last_ai.content[:50]}...")
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
    builder.add_edge("catcher", "summarizer")  # ✅ Fixed: Loop to summarizer to stop infinite DAG replanning
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
    skill_registry: Any | None = None,  # ✅ Added
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
        return builder(planner_llm, tools, planner_config, skill_registry)

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

    return builder(planner_llm, tools, planner_config, skill_registry)


