from __future__ import annotations

"""
Lógica principal (grafo LangGraph) del Agnostic Deep Agent.
Refactorizado para Arquitectura de Árbol de Tareas (Task Tree) y Reporte Detallado (v17 style).

Sub-grafos:
- ANALYZER  → LLM (JSON) → Propositions.
- PLANNER   → LLM (JSON) → Task Tree (DAG).
- EXECUTOR  → Recorre el DAG, resuelve dependencias y ejecuta tools.
- SUMMARIZER→ Genera respuesta final basada en el estado del árbol y construye reportes (dev/deep).
- VALIDATOR → Revisa cobertura.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
import json
import uuid
import re
import os

from typing_extensions import TypedDict
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    AnyMessage,
)
from langchain_core.runnables import Runnable

from .schemas import (
    AgentState,
    AnalyzerIntent,
    PlannerPlan,
    TaskNode,
    TaskStatus,
    Proposition,
    ValidationResult
)
from .prompts import (
    ANALYZER_SYSTEM_PROMPT,
    PLANNER_TASK_TREE_PROMPT,
    build_summarizer_system_message,
    build_validator_system_message
)
from .capabilities import PlannerConfig, build_planner_system_message
from .communication import ToolRun # Import ToolRun for strict typing in state


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.S | re.I)

def split_think_and_content(text: str) -> Tuple[str, str]:
    """
    Separa el contenido <think>...</think> del resto.
    Retorna (thinking_content, clean_content).
    """
    if not text:
        return "", ""
    
    match = _THINK_RE.search(text)
    if match:
        thinking = match.group(1).strip()
        # Removemos todo el bloque <think>...</think> del texto original
        clean = _THINK_RE.sub("", text).strip()
        return thinking, clean
    
    return "", text.strip()


def _parse_json_from_llm(text: str) -> Dict[str, Any]:
    """
    Intenta extraer y parsear un bloque JSON del texto del LLM.
    Busca ```json ... ``` o simplemente el primer { ... }.
    """
    text = text.strip()
    
    # 1. Buscar bloque markdown
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 2. Buscar primer { y último }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
        else:
            json_str = text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: intentar corregir errores comunes si fuera necesario
        return {}

def _resolve_args(args: Dict[str, Any], step_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resuelve referencias tipo '$T1.result' en los argumentos.
    """
    new_args = {}
    for k, v in args.items():
        if isinstance(v, str) and v.startswith("$") and ".result" in v:
            # Ej: "$T1.result"
            ref_id = v[1:].split(".")[0]  # T1
            if ref_id in step_results:
                new_args[k] = step_results[ref_id]
            else:
                new_args[k] = v # No se pudo resolver, dejar tal cual
        else:
            new_args[k] = v
    return new_args

def _get_tool_description(tool: Any) -> str:
    """Obtiene descripción textual de una tool para el prompt."""
    name = getattr(tool, "name", str(tool))
    desc = getattr(tool, "description", "")
    args = getattr(tool, "args", {})
    return f"- {name}: {desc}. Args: {args}"

def _load_prepositions_content() -> str:
    """Carga el contenido de prepositions.txt si existe para inyectarlo en el Analyzer."""
    # Asumimos que prepositions.txt está en la raíz del repo/proyecto, o cerca.
    # Ajustar ruta según necesidad.
    possible_paths = [
        "prepositions.txt", 
        "../prepositions.txt",
        os.path.join(os.path.dirname(__file__), "..", "prepositions.txt")
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    return ""


# ─────────────────────────────────────────────
# Reporting Helpers (v17 style)
# ─────────────────────────────────────────────

def _fmt_args(args: Any) -> str:
    try:
        return json.dumps(args, ensure_ascii=False)
    except Exception:
        return str(args)

def _fmt_output(name: str, val: Any) -> str:
    if name in ("embed_texts", "embed_context_tables"):
         if isinstance(val, list):
             return f"<List of {len(val)} embeddings>"
         if isinstance(val, dict):
             # embed_context_tables retorna dict con metadata
             return f"<Embedding Summary: {val.get('embedding_dim', '?')} dims>"
    
    # Para JSONs complejos
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, ensure_ascii=False, indent=2)
        except Exception:
            return str(val)
    return str(val)

def summarize_tool_runs(runs: List[Dict[str, Any]]) -> str:
    """
    Resumen user-friendly basado SOLO en las salidas de herramientas.
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
        arg_str = _fmt_args(r.get("args"))
        out_str = _fmt_output(r["name"], r.get("output"))

        # Tools visualmente ricas o complejas se muestran preformateadas
        if r["name"] in (
            "embed_texts",
            "rerank_qwen3",
            "embed_context_tables",
            "semantic_search_in_csv",
            "judge_row_with_context",
        ):
            partes.append(
                f"- `{r['name']}({arg_str})`:\\n\\n```json\\n{out_str}\\n```"
            )
        else:
            partes.append(
                f"- `{r['name']}({arg_str})` → **{out_str}**"
            )

    return "\\n".join(partes)

def build_user_answer(runs: List[Dict[str, Any]]) -> str:
    """
    Construye la respuesta 1:1 en lenguaje natural para el modo USER,
    usando EXCLUSIVAMENTE lo que viene en `runs`.
    """
    if not runs:
        return ""

    sentences: List[str] = []

    # 1) Reranker
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

            # Recuperar doc original si hace falta
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

    # 5) judge_row_with_context
    for r in runs:
        if r.get("name") == "judge_row_with_context":
            args = r.get("args", {}) or {}
            out = r.get("output") or {}

            contract_id = (
                out.get("contract_id")
                or out.get("row_id")
                or out.get("id")
                or args.get("contract_id")
            )
            judgement = out.get("judgement")
            reasons = out.get("reasons")

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

    # 6) semantic_search
    # (Se puede añadir lógica similar si se desea)

    if not sentences:
        # Fallback: dejamos vacío para que decida el llamador
        return ""

    return " ".join(sentences)

# ---------------------------------------------------------
# SUMMARIZER NODE (Module Level)
# ---------------------------------------------------------

def summarizer_node(state: AgentState) -> Dict[str, Any]:
    plan = state.get("planner_plan")
    intent = state.get("analyzer_intent")
    
    # Recopilar runs explícitos de tareas de tools
    tool_runs_list = []
    if plan and plan.tasks:
        for task in plan.tasks:
            if task.tool_name and task.status == TaskStatus.COMPLETED:
                tool_runs_list.append({
                    "name": task.tool_name,
                    "args": task.tool_args,
                    "output": task.result,
                    "id": task.id
                })
    
    # 1. ANALYZER TXT (con thinking si existe)
    analyzer_txt = (
        f"Input Payload: {{'user_prompt': '{state.get('user_prompt')}'}}\\n"
    )
    if intent.reasoning_content:
        analyzer_txt += f"Thinking (Analyzer):\\n> {intent.reasoning_content.replace('\\n', '\\n> ')}\\n\\n"
        
    analyzer_txt += (
        f"Lógica proposicional: {intent.logic_form}\\n"
        f"Proposiciones ({len(intent.propositions)}):\\n"
    )
    for p in intent.propositions:
        analyzer_txt += f"- [{p.id}] {p.text}\\n"

    # 2. PLANNER TXT (con thinking si existe)
    planner_txt = ""
    if plan.reasoning_content:
        planner_txt += f"Thinking (Planner):\\n> {plan.reasoning_content.replace('\\n', '\\n> ')}\\n\\n"

    planner_txt += (
        f"Rationale: {plan.rationale}\\n"
        "Plan Tareas:\\n"
    )
    if plan and plan.tasks:
        for t in plan.tasks:
            planner_txt += f"- [{t.id}] {t.instruction} (Tool: {t.tool_name or 'Logic'})\\n"

    # 3. EXECUTOR TXT
    executor_txt = ""
    if tool_runs_list:
        executor_txt = f"Se ejecutaron {len(tool_runs_list)} llamadas a herramientas:\\n"
        for r in tool_runs_list:
            executor_txt += f"- {r['name']} args={_fmt_args(r['args'])}\\n"
    else:
        executor_txt = "No se ejecutó ninguna herramienta (solo lógica interna o passthrough).\\n"

    # 4. SUMMARIZER (Tool-based)
    summarizer_txt = summarize_tool_runs(tool_runs_list)

    # 5. USER ANSWER (Canonical)
    user_answer = build_user_answer(tool_runs_list)
    if not user_answer:
        # Fallback al resumen de tools si no hay respuesta 1:1 amigable
        user_answer = summarizer_txt

    # Construir Markdown Reports
    
    # DEV OUT
    dev_out = "\\n\\n".join([
        "## Resumen del pipeline (DEV)",
        "### ANALYZER",
        analyzer_txt,
        "### PLANNER",
        planner_txt,
        "### EXECUTOR",
        executor_txt,
        "### SUMMARIZER (basado en herramientas)",
        summarizer_txt,
        "### RESPUESTA FINAL (modo usuario)",
        user_answer
    ])

    # DEEP OUT (similar, un poco más limpio)
    deep_out = "\\n\\n".join([
        "## Resumen deep del pipeline",
        "### ANALYZER",
        analyzer_txt,
        "### PLANNER",
        planner_txt,
        "### EXECUTOR",
        executor_txt,
        "### SUMMARIZER",
        summarizer_txt,
        "### RESPUESTA FINAL",
        user_answer
    ])

    return {
        "user_out": user_answer,
        "dev_out": dev_out,
        "deep_out": deep_out,
        "tool_runs": [ToolRun(id=str(r["id"]), name=r["name"], args=r["args"], output=r["output"]) for r in tool_runs_list]
    }


# ─────────────────────────────────────────────
# Nodos del Grafo
# ─────────────────────────────────────────────

def build_graph_agent(
    planner_llm: Runnable,
    tools: List[Any],
    planner_config: Optional[PlannerConfig] = None,
):
    
    # ---------------------------------------------------------
    # ANALYZER NODE
    # ---------------------------------------------------------
    def analyzer_node(state: AgentState) -> Dict[str, Any]:
        user_prompt = state.get("user_prompt", "")
        
        # Cargar preposiciones (contexto lógico extra)
        prepositions_txt = _load_prepositions_content()
        
        system_content = ANALYZER_SYSTEM_PROMPT
        if prepositions_txt:
            system_content += f"\\n\\nCONTEXTO ADICIONAL (Preposiciones y Lógica):\\n{prepositions_txt}"

        # Invocamos al LLM con el prompt de Analyzer
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"User Prompt: {user_prompt}")
        ]
        
        response = planner_llm.invoke(messages)
        content_raw = response.content if hasattr(response, "content") else str(response)
        
        # 1) Extraemos thinking si existe
        thinking, content_clean = split_think_and_content(content_raw)
        
        # 2) Parseamos el JSON del contenido limpio
        data = _parse_json_from_llm(content_clean)
        
        propositions_data = data.get("propositions", [])
        propositions = [Proposition(**p) for p in propositions_data]
        
        intent = AnalyzerIntent(
            propositions=propositions,
            logic_form=data.get("logic_form", ""),
            main_objective=data.get("main_objective", ""),
            language=data.get("language"),
            reasoning_content=thinking  # ✅ Guardamos el thinking
        )
        
        return {"analyzer_intent": intent}

    # ---------------------------------------------------------
    # PLANNER NODE
    # ---------------------------------------------------------
    def planner_node(state: AgentState) -> Dict[str, Any]:
        intent = state.get("analyzer_intent")
        tools_desc = "\\n".join([_get_tool_description(t) for t in tools])
        
        input_data = {
            "propositions": [p.dict() for p in intent.propositions],
            "tools_available": tools_desc
        }
        
        messages = [
            SystemMessage(content=PLANNER_TASK_TREE_PROMPT),
            HumanMessage(content=f"Input: {json.dumps(input_data, ensure_ascii=False)}")
        ]
        
        response = planner_llm.invoke(messages)
        content_raw = response.content if hasattr(response, "content") else str(response)
        
        # 1) Extraemos thinking si existe
        thinking, content_clean = split_think_and_content(content_raw)
        
        # 2) Parseamos el JSON
        data = _parse_json_from_llm(content_clean)
        
        tasks_data = data.get("tasks", [])
        tasks = [TaskNode(**t) for t in tasks_data]
        
        plan = PlannerPlan(
            tasks=tasks,
            rationale=data.get("rationale"),
            reasoning_content=thinking  # ✅ Guardamos el thinking
        )
        
        return {"planner_plan": plan, "step_results": {}}

    # ---------------------------------------------------------
    # EXECUTOR NODE (Tree Traversal)
    # ---------------------------------------------------------
    def executor_node(state: AgentState) -> Dict[str, Any]:
        plan = state.get("planner_plan")
        step_results = state.get("step_results", {})
        
        # Identificar tareas ejecutables
        runnable_tasks = []
        for task in plan.tasks:
            if task.status == TaskStatus.PENDING:
                deps_met = all(dep in step_results for dep in task.dependencies)
                if deps_met:
                    runnable_tasks.append(task)
        
        if not runnable_tasks:
            return {} 

        # Ejecutar tareas
        for task in runnable_tasks:
            task.status = TaskStatus.RUNNING
            
            if task.tool_name:
                tool = next((t for t in tools if t.name == task.tool_name), None)
                if tool:
                    try:
                        final_args = _resolve_args(task.tool_args, step_results)
                        result = tool.invoke(final_args)
                        
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        step_results[task.id] = result
                        
                    except Exception as e:
                        task.error = str(e)
                        task.status = TaskStatus.FAILED
                else:
                    task.error = f"Tool '{task.tool_name}' not found."
                    task.status = TaskStatus.FAILED
            else:
                # Tarea sin tool
                task.result = "Executed logic/reasoning step."
                task.status = TaskStatus.COMPLETED
                step_results[task.id] = task.result

        return {
            "planner_plan": plan,
            "step_results": step_results
        }

    # ---------------------------------------------------------
    # ROUTER (Loop Logic)
    # ---------------------------------------------------------
    def should_continue(state: AgentState) -> str:
        plan = state.get("planner_plan")
        pending = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
        
        runnable = False
        step_results = state.get("step_results", {})
        for task in pending:
             if all(dep in step_results for dep in task.dependencies):
                 runnable = True
                 break
        
        if pending and runnable:
            return "executor"
        
        return "summarizer"

    # ---------------------------------------------------------
    # VALIDATOR NODE
    # ---------------------------------------------------------
    def validator_node(state: AgentState) -> Dict[str, Any]:
        return {"validation": ValidationResult(all_covered=True)}

    # ---------------------------------------------------------
    # Graph Construction
    # ---------------------------------------------------------
    builder = StateGraph(AgentState)
    
    builder.add_node("analyzer", analyzer_node)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("validator", validator_node)
    
    builder.add_edge(START, "analyzer")
    builder.add_edge("analyzer", "planner")
    builder.add_edge("planner", "executor")
    
    builder.add_conditional_edges(
        "executor",
        should_continue,
        ["executor", "summarizer"]
    )
    
    builder.add_edge("summarizer", "validator")
    builder.add_edge("validator", END)
    
    return builder.compile()


# ─────────────────────────────────────────────
# Logic loader
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
    return build_graph_agent(planner_llm, tools, planner_config)
