from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import os

import yaml
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    AnyMessage,
)

from .capabilities import PlannerConfig, build_planner_llm
from .logic import load_logic, State
from .communication import (
    AgentInput,
    AgentOutput,
    AgentView,
    AgentSummary,
    ToolRun,
)
from .tools import get_default_tools  # ✅ catálogo global de tools
from .memory import read_memory, write_memory  # ✅ memoria multi-nivel (in-memory)
from .knowledge import (  # ✅ KBs externas/tabulares
    KnowledgeBase,
    get_default_context,
    select_knowledge_bases,
    build_kb_from_setup,
)


from .skills import SkillRegistry  # ✅ registro de skills
from .app.turn_service import TurnService
from .app.errors import TurnExecutionError

class Agent:
    """
    Agente agnóstico sobre LangGraph + OpenAI-compatible LLM.
    ...
    """

    def __init__(
        self,
        graph_app: Any,
        planner_config: PlannerConfig,
        tools: List[Any],
        *,
        setup_path: Optional[str] = None,
        setup_config: Optional[Dict[str, Any]] = None,
        memory_cfg: Optional[Dict[str, Any]] = None,
        knowledge_bases: Optional[List[KnowledgeBase]] = None,
        context_tables: Optional[List[str]] = None,
        context_cfg: Optional[Dict[str, Any]] = None,
        skill_registry: Optional[SkillRegistry] = None,
    ) -> None:
        self.graph_app = graph_app
        self.planner_config = planner_config
        self.tools = tools
        self.skill_registry = skill_registry

        # Initialize TurnService with all necessary dependencies
        self.turn_service = TurnService(
            graph_app=graph_app,
            knowledge_bases=knowledge_bases if knowledge_bases is not None else get_default_context(),
            memory_cfg=memory_cfg or {},
            context_tables=context_tables or [],
            context_cfg=context_cfg or {},
            setup_path=setup_path,
            setup_config=setup_config,
        )

        # Backward compatibility for attributes accessed directly
        self.setup_path = setup_path
        self.setup_config = setup_config or {}
        self.memory_cfg = memory_cfg or {}
        self.knowledge_bases = self.turn_service.knowledge_bases
        self.context_tables = self.turn_service.context_tables
        self.context_cfg = self.turn_service.context_cfg

    # ------------------------------------------------------------------
    # Helpers de setup.yaml
    # ------------------------------------------------------------------
    @staticmethod
    def _load_setup_config(
        setup_path: Optional[Union[str, Path]],
    ) -> Tuple[Optional[Path], Dict[str, Any]]:
        """
        Intenta cargar setup.yaml (o el path que se pase).

        Orden de resolución:
          1) setup_path explícito (argumento).
          2) AGENT_SETUP_PATH en variables de entorno.
        """
        cfg: Dict[str, Any] = {}

        path_obj: Optional[Path] = None
        if isinstance(setup_path, (str, Path)):
            path_obj = Path(setup_path)
        else:
            env_path = os.getenv("AGENT_SETUP_PATH")
            if env_path:
                path_obj = Path(env_path)

        if path_obj is None:
            return None, cfg

        if not path_obj.is_file():
            # No levantamos excepción para no romper en Colab si no está el archivo
            print(f"[Agent] ⚠️ setup.yaml no encontrado en: {path_obj}. Se usarán defaults.")
            return path_obj, cfg

        try:
            with path_obj.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                print(f"[Agent] ⚠️ setup.yaml no tiene formato dict en {path_obj}. Ignorando.")
                return path_obj, {}
            cfg = data
        except Exception as e:
            print(f"[Agent] ⚠️ Error leyendo setup.yaml ({path_obj}): {e!r}")
            cfg = {}

        return path_obj, cfg

    @staticmethod
    def _apply_model_env_from_setup(setup_cfg: Dict[str, Any]) -> None:
        """
        Opcional: aplica variables de entorno para modelos / endpoints
        a partir de setup.yaml, sin pisar lo que ya esté definido.
        """
        models_cfg = setup_cfg.get("models") or {}

        # LLM
        llm_cfg = models_cfg.get("llm") or {}
        llm_api_base = llm_cfg.get("api_base")
        llm_served_name = llm_cfg.get("served_name")

        if llm_api_base and "VLLM_API_BASE" not in os.environ:
            os.environ["VLLM_API_BASE"] = str(llm_api_base)
        if llm_api_base and "VLLM_LLM_API_BASE" not in os.environ:
            os.environ["VLLM_LLM_API_BASE"] = str(llm_api_base)
        if llm_served_name and "LLM_SERVED_NAME" not in os.environ:
            os.environ["LLM_SERVED_NAME"] = str(llm_served_name)

        # Embeddings
        emb_cfg = models_cfg.get("emb") or {}
        emb_api_base = emb_cfg.get("api_base")
        emb_served_name = emb_cfg.get("served_name")

        if emb_api_base and "VLLM_EMB_API_BASE" not in os.environ:
            os.environ["VLLM_EMB_API_BASE"] = str(emb_api_base)
        if emb_served_name and "EMB_SERVED_NAME" not in os.environ:
            os.environ["EMB_SERVED_NAME"] = str(emb_served_name)

        # Reranker
        rerank_cfg = models_cfg.get("rerank") or {}
        rerank_api_base = rerank_cfg.get("api_base")
        rerank_served_name = rerank_cfg.get("served_name")

        if rerank_api_base and "VLLM_RERANK_API_BASE" not in os.environ:
            os.environ["VLLM_RERANK_API_BASE"] = str(rerank_api_base)
        if rerank_served_name and "RERANK_SERVED_NAME" not in os.environ:
            os.environ["RERANK_SERVED_NAME"] = str(rerank_served_name)

        # Clave dummy para OpenAI-compatible (vLLM la ignora pero la requiere)
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "EMPTY"

    @staticmethod
    def _build_kb_from_setup(setup_cfg: Dict[str, Any]) -> List[KnowledgeBase]:
        """
        Construye la lista de KnowledgeBase a partir de setup.yaml usando
        el helper genérico de context.build_kb_from_setup.

        Si no hay nada en el YAML, cae en get_default_context().
        """
        knowledge_list = build_kb_from_setup(setup_cfg)
        if not knowledge_list:
            knowledge_list = get_default_context()
        return knowledge_list

    @staticmethod
    def _resolve_context_tables(
        setup_cfg: Dict[str, Any],
        explicit_context_tables: Optional[List[str]],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Resuelve tablas de contexto (CSV) y config de contexto.

        Prioridad:
          1) context_tables explícito pasado a Agent.init(...)
          2) setup.yaml:
               context:
                 tables: [...]
               # o bien, compat:
               context_tables: [...]
        """
        context_cfg: Dict[str, Any] = setup_cfg.get("context") or {}

        yaml_tables = context_cfg.get("tables") or setup_cfg.get("context_tables") or []
        if isinstance(yaml_tables, str):
            yaml_tables = [yaml_tables]
        yaml_tables = [str(p) for p in yaml_tables] if isinstance(yaml_tables, list) else []

        if explicit_context_tables is not None:
            final_tables = list(explicit_context_tables)
        else:
            final_tables = yaml_tables

        return final_tables, context_cfg

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def init(
        cls,
        config_or_setup: Optional[Union[PlannerConfig, str, os.PathLike]] = None,
        tools: Optional[List[Any]] = None,
        *,
        context_tables: Optional[List[str]] = None,
        skills_dir: Optional[str] = None,
    ) -> "Agent":
        """
        Construye un Agent listo para usar.
        """
        # 1) Resolver si el primer parámetro es un PlannerConfig o un path
        setup_path: Optional[Union[str, Path]] = None
        planner_cfg: Optional[PlannerConfig] = None

        if isinstance(config_or_setup, PlannerConfig):
            planner_cfg = config_or_setup
        elif isinstance(config_or_setup, (str, os.PathLike)):
            setup_path = config_or_setup

        # 2) Cargar setup.yaml (si existe)
        setup_path_resolved, setup_cfg = cls._load_setup_config(setup_path)

        # 3) Aplicar envs de modelos (si vienen en setup.yaml)
        if setup_cfg:
            cls._apply_model_env_from_setup(setup_cfg)

        # 4) PlannerConfig: o bien el explícito, o bien override desde YAML
        if planner_cfg is not None:
            cfg = planner_cfg
        else:
            cfg = PlannerConfig()
            planner_section = setup_cfg.get("planner") or {}
            # Sobrescribimos sólo campos conocidos (tolerante)
            for key, value in planner_section.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

        # 5) Tools: prioridad al parámetro explícito; si no, leer de setup.yaml
        if tools is not None:
            tools_list = tools
        else:
            tools_section = setup_cfg.get("tools") or {}
            enabled_names = tools_section.get("enabled")
            tools_list = get_default_tools(enabled_names=enabled_names)

        # 5.5) Inicializar Skills
        # Intentar buscar 'skills' dir en setup.yaml o argumento o default
        skills_path = skills_dir
        if not skills_path:
            skills_section = setup_cfg.get("skills") or {}
            skills_path = skills_section.get("path")
        
        if not skills_path:
             # Default: agnostic_agent/skills sibling to this file
             base_dir = os.path.dirname(__file__)
             skills_path = os.path.join(base_dir, "skills")
        
        skill_registry = SkillRegistry(skills_path)

        # 6) LLM planner bindeado a las tools
        planner_llm = build_planner_llm(cfg)
        planner_llm = planner_llm.bind_tools(tools_list)

        # 7) Construir grafo principal
        graph_app = load_logic(
            planner_llm=planner_llm,
            tools=tools_list,
            planner_config=cfg,
            skill_registry=skill_registry,
        )

        # 8) Config de memoria desde setup.yaml (si existe)
        memory_cfg = setup_cfg.get("memory") or {}

        # 9) Knowledge Bases (tabulares / vectores / APIs) desde setup.yaml
        knowledge_list = cls._build_kb_from_setup(setup_cfg)

        # 10) Tablas de contexto (parametrías, abreviaturas, etc.)
        final_context_tables, context_cfg = cls._resolve_context_tables(
            setup_cfg=setup_cfg,
            explicit_context_tables=context_tables,
        )

        return cls(
            graph_app=graph_app,
            planner_config=cfg,
            tools=tools_list,
            setup_path=str(setup_path_resolved) if setup_path_resolved else None,
            setup_config=setup_cfg,
            memory_cfg=memory_cfg,
            knowledge_bases=knowledge_list,
            context_tables=final_context_tables,
            context_cfg=context_cfg,
            skill_registry=skill_registry,
        )

    # ------------------------------------------------------------------
    # API público (Delegado a TurnService)
    # ------------------------------------------------------------------
    def run_turn(
        self,
        user_input: Union[str, Dict[str, Any], AgentInput],
    ) -> Dict[str, Any]:
        """
        Ejecuta un turno de conversación delegando al TurnService.
        """
        return self.turn_service.run_turn(user_input)

    # ------------------------------------------------------------------
    # Extras útiles para debugging
    # ------------------------------------------------------------------
    @property
    def last_state(self) -> Optional[Dict[str, Any]]:
        """Devuelve el último estado crudo del grafo (solo lectura)."""
        return self.turn_service.last_state

    def reset_session(self) -> None:
        """Resetea el estado interno de conversación (no borra memoria global)."""
        self.turn_service.reset_session()
