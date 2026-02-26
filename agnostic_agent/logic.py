from __future__ import annotations

"""
Logica principal (grafo LangGraph) del Agnostic Deep Agent.

Sub-grafos actuales:
- ANALYZER  a descompone el prompt (rule-based sencillo por ahora).
- PLANNER   a usa Planner LLM (OpenAI-compatible) para generar tool_calls.
- EXECUTOR  a ejecuta tools reales (LangChain tools).
- CATCHER   a normaliza las salidas de tools a una lista de runs.
- SUMMARIZERa construye:
    - respuesta final en modo usuario (user_answer),
    - resumen tAcnico del pipeline (para vistas deep/dev).
- VALIDATOR a revisa si la respuesta parece cubrir todo lo pedido.

Notas:
- Este mA3dulo sigue usando TypedDict; todavAa no esta cableado
  a los modelos Pydantic de `schemas.py`.
- Ya integra memoria y knowledge_names en el planner, y deja
  dev_out / deep_out / user_out en el estado.
- EstA pensado para casos donde el agente cruza:
    * una tabla de atributos (input A, p.ej. filas de contratos),
    * con tablas de contexto (input B, p.ej. parametrAas y
      diccionarios de abreviaturas/definiciones),
    * y, opcionalmente, documentos (OCR de contratos) vAa tools
      como context_search_in_csv + rerank_docs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    AnyMessage,
    SystemMessage,
)

from .capabilities import PlannerConfig
from .graph.builders import (
    compile_agent_graph,
    route_from_planner as route_from_planner_shared,
)
from .graph.validator_node import execute_validator_node
from .graph.summarizer_node import execute_summarizer_node
from .graph.catcher_node import execute_catcher_node
from .graph.executor_node import execute_executor_node
from .graph.analyzer_node import execute_analyzer_node
from .graph.planner_node import execute_planner_node
from .graph.contracts import PlannerTrajectory, State
from .graph.runtime_utils import (
    _canonical_tool_name,
    _coerce_content_str,
    _decode_tool_content,
    _env_flag,
    _extract_tool_calls_from_jsonish_text,
    _extract_top_level_json_objects,
    _is_ai_with_tool_calls,
    _is_pipeline_internal_ai,
    _is_placeholder_subquery,
    _json_default,
    _normalize_toolcalls_list,
    _resolve_effective_skills,
    _sanitize_subquery_text,
    _to_jsonable,
    build_user_answer_from_runs,
    extract_tool_calls,
    find_last_assistant_real,
    is_technical_answer,
    strip_think,
    summarize_tool_runs,
    summarize_tool_runs_compact,
)



def build_graph_agent(
    planner_llm,
    tools: List[Any],
    planner_config: PlannerConfig | None = None,
    skill_registry: Any | None = None,  # a... Recibimos el registro
):
    """
    Grafo:

        START a ANALYZER a PLANNER
                      aa(tool_calls)a EXECUTOR a CATCHER a SUMMARIZER a VALIDATOR a END
                      aaaaaaaaaaaaaaa SUMMARIZER a VALIDATOR a END
    """
    cfg = planner_config or PlannerConfig()

    # ANALYZER (LLM-based with Strict JSON)
    def analyzer_node(state: State) -> Dict[str, Any]:
        return execute_analyzer_node(
            state,
            tools=tools,
            cfg=cfg,
            planner_llm=planner_llm,
            skill_registry=skill_registry,
            ai_message_type=AIMessage,
            human_message_type=HumanMessage,
            system_message_type=SystemMessage,
            coerce_content_str=_coerce_content_str,
            sanitize_subquery_text=_sanitize_subquery_text,
            extract_top_level_json_objects=_extract_top_level_json_objects,
            is_placeholder_subquery=_is_placeholder_subquery,
        )

    def planner_node(state: State) -> Dict[str, Any]:
        return execute_planner_node(
            state,
            tools=tools,
            cfg=cfg,
            planner_llm=planner_llm,
            skill_registry=skill_registry,
            ai_message_type=AIMessage,
            human_message_type=HumanMessage,
            system_message_type=SystemMessage,
            planner_trajectory_type=PlannerTrajectory,
            resolve_effective_skills=_resolve_effective_skills,
            is_pipeline_internal_ai=_is_pipeline_internal_ai,
            is_ai_with_tool_calls=_is_ai_with_tool_calls,
            strip_think=strip_think,
            normalize_toolcalls_list=_normalize_toolcalls_list,
            extract_tool_calls_from_jsonish_text=_extract_tool_calls_from_jsonish_text,
            coerce_content_str=_coerce_content_str,
            canonical_tool_name=_canonical_tool_name,
        )


    # EXECUTOR
    def executor_node(state: State) -> Dict[str, Any]:
        return execute_executor_node(
            state,
            tools=tools,
            ai_message_type=AIMessage,
            tool_message_type=ToolMessage,
            extract_tool_calls=extract_tool_calls,
            canonical_tool_name=_canonical_tool_name,
            to_jsonable=_to_jsonable,
            json_default=_json_default,
        )

    # CATCHER
    def catcher_node(state: State) -> Dict[str, Any]:
        return execute_catcher_node(
            state,
            extract_tool_calls=extract_tool_calls,
            decode_tool_content=_decode_tool_content,
            to_jsonable=_to_jsonable,
        )

    # SUMMARIZER
    def summarizer_node(state: State) -> Dict[str, Any]:
        return execute_summarizer_node(
            state,
            skill_registry=skill_registry,
            tools=tools,
            cfg=cfg,
            planner_llm=planner_llm,
            resolve_effective_skills=_resolve_effective_skills,
            json_default=_json_default,
            summarize_tool_runs=summarize_tool_runs,
            summarize_tool_runs_compact=summarize_tool_runs_compact,
            build_user_answer_from_runs=build_user_answer_from_runs,
            is_technical_answer=is_technical_answer,
            find_last_assistant_real=find_last_assistant_real,
            extract_tool_calls=extract_tool_calls,
            coerce_content_str=_coerce_content_str,
            strip_think=strip_think,
        )

    # VALIDATOR (heurAstica simple, preparada para LLM en el futuro)
    # VALIDATOR (heurAstica simple, preparada para LLM en el futuro)
    def validator_node(state: State) -> Dict[str, Any]:
        return execute_validator_node(
            state,
            skill_registry=skill_registry,
            resolve_effective_skills=_resolve_effective_skills,
            is_placeholder_subquery=_is_placeholder_subquery,
            env_flag=_env_flag,
            extract_top_level_json_objects=_extract_top_level_json_objects,
            find_last_assistant_real=find_last_assistant_real,
            coerce_content_str=_coerce_content_str,
            strip_think=strip_think,
            build_user_answer_from_runs=build_user_answer_from_runs,
            is_technical_answer=is_technical_answer,
        )

    # Router (Updated Debug)
    def route_from_planner(state: State) -> str:
        return route_from_planner_shared(
            state,
            ai_message_type=AIMessage,
            extract_tool_calls=extract_tool_calls,
        )

    return compile_agent_graph(
        State,
        analyzer_node=analyzer_node,
        planner_node=planner_node,
        executor_node=executor_node,
        catcher_node=catcher_node,
        summarizer_node=summarizer_node,
        validator_node=validator_node,
        route_from_planner_fn=route_from_planner,
    )



# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# Logic loader (registro de grafos)
# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

@dataclass
class LogicConfig:
    module: str = "agnostic_agent.logic"
    builder_fn: str = "build_graph_agent"


def load_logic(
    planner_llm: Any,
    tools: List[Any],
    planner_config: Optional[PlannerConfig] = None,
    logic_config: Optional[LogicConfig] = None,
    skill_registry: Any | None = None,  # a... Added
) -> Any:
    """
    Carga y ejecuta la funciA3n builder que construye el grafo del agente.

    Por defecto usa este mismo mA3dulo:
        agnostic_agent.logic.build_graph_agent
    """
    cfg = logic_config or LogicConfig()

    if cfg.module == "agnostic_agent.logic":
        builder: Callable[..., Any] = globals().get(cfg.builder_fn)  # type: ignore[assignment]
        if builder is None or not callable(builder):
            raise AttributeError(
                f"No se encontrA3 funciA3n builder '{cfg.builder_fn}' en agnostic_agent.logic."
            )
        return builder(planner_llm, tools, planner_config, skill_registry)

    import importlib

    try:
        mod = importlib.import_module(cfg.module)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"No se pudo importar el mA3dulo de lA3gica '{cfg.module}'."
        ) from e

    builder = getattr(mod, cfg.builder_fn, None)
    if builder is None or not callable(builder):
        raise AttributeError(
            f"El mA3dulo '{cfg.module}' no tiene una funciA3n callable '{cfg.builder_fn}'."
        )

    return builder(planner_llm, tools, planner_config, skill_registry)


