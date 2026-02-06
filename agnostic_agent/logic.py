from __future__ import annotations

"""
Lógica principal (grafo LangGraph) del Agnostic Deep Agent.
Refactorizado para Arquitectura de Árbol de Tareas (Task Tree).

Sub-grafos:
- ANALYZER  → LLM (JSON) → Propositions.
- PLANNER   → LLM (JSON) → Task Tree (DAG).
- EXECUTOR  → Recorre el DAG, resuelve dependencias y ejecuta tools.
- SUMMARIZER→ Genera respuesta final basada en el estado del árbol.
- VALIDATOR → Revisa cobertura.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
import json
import uuid
import re

from typing_extensions import TypedDict
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


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

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
        
        # Invocamos al LLM con el prompt de Analyzer
        messages = [
            SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
            HumanMessage(content=f"User Prompt: {user_prompt}")
        ]
        
        response = planner_llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        
        data = _parse_json_from_llm(content)
        
        propositions_data = data.get("propositions", [])
        propositions = [Proposition(**p) for p in propositions_data]
        
        intent = AnalyzerIntent(
            propositions=propositions,
            logic_form=data.get("logic_form", ""),
            main_objective=data.get("main_objective", ""),
            language=data.get("language")
        )
        
        return {"analyzer_intent": intent}

    # ---------------------------------------------------------
    # PLANNER NODE
    # ---------------------------------------------------------
    def planner_node(state: AgentState) -> Dict[str, Any]:
        intent = state.get("analyzer_intent")
        tools_desc = "\n".join([_get_tool_description(t) for t in tools])
        
        input_data = {
            "propositions": [p.dict() for p in intent.propositions],
            "tools_available": tools_desc
        }
        
        messages = [
            SystemMessage(content=PLANNER_TASK_TREE_PROMPT),
            HumanMessage(content=f"Input: {json.dumps(input_data, ensure_ascii=False)}")
        ]
        
        response = planner_llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        
        data = _parse_json_from_llm(content)
        
        tasks_data = data.get("tasks", [])
        tasks = [TaskNode(**t) for t in tasks_data]
        
        plan = PlannerPlan(
            tasks=tasks,
            rationale=data.get("rationale")
        )
        
        return {"planner_plan": plan, "step_results": {}}

    # ---------------------------------------------------------
    # EXECUTOR NODE (Tree Traversal)
    # ---------------------------------------------------------
    def executor_node(state: AgentState) -> Dict[str, Any]:
        plan = state.get("planner_plan")
        step_results = state.get("step_results", {})
        
        # Identificar tareas ejecutables
        # Una tarea es ejecutable si:
        # 1. Estado es PENDING
        # 2. Todas sus dependencias están en step_results (o completadas)
        
        runnable_tasks = []
        for task in plan.tasks:
            if task.status == TaskStatus.PENDING:
                deps_met = all(dep in step_results for dep in task.dependencies)
                if deps_met:
                    runnable_tasks.append(task)
        
        if not runnable_tasks:
            return {} # Nada que hacer en esta vuelta

        # Ejecutar tareas (secuencialmente por ahora por simplicidad, podría ser paralelo)
        for task in runnable_tasks:
            task.status = TaskStatus.RUNNING
            
            # Resolver tool
            if task.tool_name:
                tool = next((t for t in tools if t.name == task.tool_name), None)
                if tool:
                    try:
                        # Resolver argumentos
                        final_args = _resolve_args(task.tool_args, step_results)
                        # Ejecutar
                        result = tool.invoke(final_args)
                        
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        step_results[task.id] = result
                        
                    except Exception as e:
                        task.error = str(e)
                        task.status = TaskStatus.FAILED
                        # Opcional: Fail fast o continuar? Continuamos.
                else:
                    task.error = f"Tool '{task.tool_name}' not found."
                    task.status = TaskStatus.FAILED
            else:
                # Tarea sin tool (lógica, razonamiento puro)
                # La marcamos completada, resultado es su instrucción/thought?
                # O quizás requerimos intervención del LLM?
                # Por ahora, asumimos que es 'passthrough' o hecha.
                task.result = "Executed logic/reasoning step."
                task.status = TaskStatus.COMPLETED
                step_results[task.id] = task.result

        return {
            "planner_plan": plan, # Actualiza el estado de las tareas
            "step_results": step_results
        }

    # ---------------------------------------------------------
    # ROUTER (Loop Logic)
    # ---------------------------------------------------------
    def should_continue(state: AgentState) -> str:
        plan = state.get("planner_plan")
        
        # Verificar si quedan tareas pendientes
        pending = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
        
        # Verificar si falló todo o si hay progreso posible
        # Si hay pendientes pero sus dependencias fallaron, estamos en deadlock.
        # Check deadlock:
        runnable = False
        step_results = state.get("step_results", {})
        for task in pending:
             if all(dep in step_results for dep in task.dependencies):
                 runnable = True
                 break
        
        if pending and runnable:
            return "executor" # Volver a ejecutar
        
        return "summarizer" # Terminamos (o deadlock)

    # ---------------------------------------------------------
    # SUMMARIZER NODE
    # ---------------------------------------------------------
    def summarizer_node(state: AgentState) -> Dict[str, Any]:
        plan = state.get("planner_plan")
        intent = state.get("analyzer_intent")
        
        # Construir contexto para el LLM sumarizador
        context_str = f"Objetivo: {intent.main_objective}\n\nEjecución:\n"
        for task in plan.tasks:
            status_icon = "✅" if task.status == TaskStatus.COMPLETED else "❌" if task.status == TaskStatus.FAILED else "⏳"
            context_str += f"{status_icon} [{task.id}] {task.instruction}\n"
            if task.result:
                context_str += f"   Resultado: {str(task.result)[:500]}\n" # Truncar resultados largos
            if task.error:
                context_str += f"   Error: {task.error}\n"

        # Prompt simple para respuesta final (podríamos usar los de prompts.py adaptados)
        sys_msg = build_summarizer_system_message("user")
        msgs = [
            sys_msg,
            HumanMessage(content=f"Genera la respuesta final basada en esta ejecución:\n{context_str}")
        ]
        
        response = planner_llm.invoke(msgs)
        final_text = response.content if hasattr(response, "content") else str(response)
        
        return {"user_out": final_text}

    # ---------------------------------------------------------
    # VALIDATOR NODE
    # ---------------------------------------------------------
    def validator_node(state: AgentState) -> Dict[str, Any]:
        # Implementación simple por ahora
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
    # (Mantener implementación original de carga dinámica si se desea, 
    #  o simplificar llamando directo a build_graph_agent)
    cfg = logic_config or LogicConfig()
    return build_graph_agent(planner_llm, tools, planner_config)
