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

        # Panel de control cargado desde setup.yaml (si existe)
        self.setup_path: Optional[str] = setup_path
        self.setup_config: Dict[str, Any] = setup_config or {}

        # Config de memoria (session / short_term / long_term)
        self.memory_cfg: Dict[str, Any] = memory_cfg or {}

        # KBs registradas (tabulares, vectores, APIs, etc.)
        self.knowledge_bases: List[KnowledgeBase] = (
            knowledge_bases if knowledge_bases is not None else get_default_context()
        )

        # Tablas CSV de contexto (parametrías, abreviaturas/definiciones, etc.)
        self.context_tables: List[str] = context_tables or []

        # Config de contexto crudo desde setup.yaml (opcional)
        self.context_cfg: Dict[str, Any] = context_cfg or {}

        # Registro de Skills
        self.skill_registry = skill_registry

        # Estado de conversación multi-turn (historial reducido)
        self._state: Optional[Dict[str, Any]] = None

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

        Patrones soportados:

            agent = Agent.init()
            agent = Agent.init("setup.yaml")
            agent = Agent.init(setup_path)  # PathLike
            agent = Agent.init(PlannerConfig(...))
            agent = Agent.init("setup.yaml", context_tables=[...])

        NOTA:
        - Si pasas PlannerConfig → se ignora el bloque `planner` de setup.yaml.
        - Si quieres combinar ambas cosas, construye tú el PlannerConfig
          leyendo el YAML y pásalo aquí.
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
    # Helpers internos
    # ------------------------------------------------------------------
    def _coerce_input(
        self,
        user_input: Union[str, Dict[str, Any], AgentInput],
    ) -> AgentInput:
        """
        Normaliza la entrada a AgentInput:

        - str  -> AgentInput(user_prompt=...)
        - dict -> AgentInput(**dict)
        - AgentInput -> se respeta tal cual
        """
        if isinstance(user_input, AgentInput):
            return user_input
        if isinstance(user_input, dict):
            return AgentInput(**user_input)
        return AgentInput(user_prompt=str(user_input))

    def _clean_prev_messages(self) -> List[AnyMessage]:
        """
        Limpia el historial previo para evitar ToolMessages gigantes en turnos futuros.
        """
        msgs: List[AnyMessage] = []
        if self._state is None:
            return msgs

        for m in self._state.get("messages", []):
            # ❌ No arrastramos ToolMessages con JSONs enormes
            if isinstance(m, ToolMessage):
                continue
            msgs.append(m)
        return msgs

    def _build_deep_text(
        self,
        summary_obj: Optional[AgentSummary],
    ) -> str:
        """
        Construye la vista 'deep' (resumen por sección) a partir de AgentSummary.

        - No incluye el raw_state completo.
        - Es más compacta que la traza dev_out, pero sigue seccionada.
        """
        if summary_obj is None:
            # Si por alguna razón no hay summary, devolvemos cadena vacía;
            # el caller puede hacer fallback.
            return ""

        parts: List[str] = ["## Resumen deep del pipeline"]

        if summary_obj.analyzer:
            parts.append("### ANALYZER\n" + summary_obj.analyzer)
        if summary_obj.planner:
            parts.append("### PLANNER\n" + summary_obj.planner)
        if summary_obj.executor:
            parts.append("### EXECUTOR\n" + summary_obj.executor)
        if summary_obj.catcher:
            parts.append("### CATCHER\n" + summary_obj.catcher)
        if summary_obj.summarizer:
            parts.append("### SUMMARIZER (basado en herramientas)\n" + summary_obj.summarizer)
        if summary_obj.final_answer:
            parts.append("### RESPUESTA FINAL\n" + summary_obj.final_answer)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # API público
    # ------------------------------------------------------------------
    def run_turn(
        self,
        user_input: Union[str, Dict[str, Any], AgentInput],
    ) -> Dict[str, Any]:
        """
        Ejecuta un turno de conversación.

        Devuelve SIEMPRE un dict:

            {
              "dev_out": {
                  "final_answer": "...pipeline markdown (traza completa)...",
                  "summary": {...},
                  "tool_runs": [...],
                  "raw_state": {...}   # estado crudo del grafo
              },
              "deep_out": {
                  "final_answer": "...resumen por sección...",
                  "summary": {...},
                  "tool_runs": [...],
                  "raw_state": {}      # vista ligera
              },
              "user_out": {
                  "final_answer": "...respuesta para usuario (1:1)...",
                  "summary": {...},
                  "tool_runs": [...],
                  "raw_state": {}
              }
            }
        """
        agent_in = self._coerce_input(user_input)

        # Texto canónico del prompt (user_prompt > user_text como fallback)
        prompt_text = (
            getattr(agent_in, "user_prompt", None)
            or getattr(agent_in, "user_text", None)
            or ""
        )

        # -------------------------
        # 0) Resolver session_id / user_id / knowledge_names
        # -------------------------
        session_id = agent_in.session_id or "default"
        user_id = None
        if agent_in.metadata:
            user_id = agent_in.metadata.get("user_id")

        # Force default to ALL registered KBs if none specified
        # This fixes the issue where Planner sees "(No knowledge bases active)"
        # when running from Streamlit without explicit knowledge_names in payload.
        knowledge_names = agent_in.knowledge_names
        if not knowledge_names:
            knowledge_names = [kb.name for kb in self.knowledge_bases]

        # Seleccionar KBs activas para este turno (si knowledge_names está vacío, usamos todas)
        knowledge_selected = select_knowledge_bases(knowledge_names, self.knowledge_bases)

        # -------------------------
        # 1) Leer memoria para esta sesión
        # -------------------------
        memory_context = read_memory(session_id=session_id)

        # 2) Construir estado de entrada al grafo
        prev_messages = self._clean_prev_messages()
        state_in: State = {
            "messages": prev_messages + [HumanMessage(content=prompt_text)],
            "analyzer": None,
            "planner_trajs": [],
            "executor_steps": [],
            "tool_runs": [],
            "summary": None,
            "pipeline_summary": None,
            # Campos extra (no tipados en State, pero soportados por TypedDict total=False)
            "user_prompt": prompt_text,
            "session_id": session_id,
            "user_id": user_id,
            "setup_path": self.setup_path or "",
            "setup_config": self.setup_config,  # 👈 YAML completo (por si lo necesita el grafo)
            "knowledge_names": knowledge_names,
            "kb_all": [kb.__dict__ for kb in self.knowledge_bases],
            "knowledge_selected": [kb.__dict__ for kb in knowledge_selected],
            "memory_context": memory_context,
            # Tablas de contexto (para semantic_search_in_csv en el grafo)
            "context_tables": list(self.context_tables),
            "context_cfg": self.context_cfg,
        }

        # 3) Invocar grafo
        out_state: State = self.graph_app.invoke(state_in)
        self._state = out_state  # guardamos para multi-turn

        # 4) Extraer último AIMessage (por si no vienen campos dev_out/deep_out en state)
        ai_messages = [
            m for m in out_state.get("messages", []) if isinstance(m, AIMessage)
        ]
        last_ai = ai_messages[-1] if ai_messages else None
        last_ai_text = last_ai.content if last_ai is not None else ""

        # 5) Campos de texto que el grafo ya pudo haber dejado en el estado
        dev_text_state = out_state.get("dev_out")  # type: ignore[assignment]
        deep_text_state = out_state.get("deep_out")  # type: ignore[assignment]
        user_text_state = out_state.get("user_out")  # type: ignore[assignment]

        # 6) Summary estructurado (con final_answer "para usuario")
        raw_summary: Dict[str, Any] = (
            out_state.get("pipeline_summary")  # type: ignore[arg-type]
            or out_state.get("summary")  # type: ignore[arg-type]
            or {}
        )
        if raw_summary:
            summary_obj = AgentSummary(**raw_summary)
            summary_user_answer = summary_obj.final_answer or ""
        else:
            summary_obj = None
            summary_user_answer = ""

        # 7) Tool runs normalizados (para las tres vistas)
        raw_runs = out_state.get("tool_runs", []) or []
        tool_runs: List[ToolRun] = []
        for r in raw_runs:
            tool_runs.append(
                ToolRun(
                    id=str(r.get("id", "")),
                    name=str(r.get("name", "")),
                    args=r.get("args", {}),
                    output=r.get("output"),
                )
            )

        # 8) Resolver textos finales por vista con prioridades claras
        # USER:
        #   1) user_text_state (si el grafo lo puso)
        #   2) summary_user_answer (si existe en AgentSummary)
        #   3) last_ai_text (último mensaje de la traza)
        final_user = (
            (user_text_state or "").strip()
            or summary_user_answer.strip()
            or last_ai_text.strip()
        )

        # DEEP:
        #   1) deep_text_state (del grafo)
        #   2) _build_deep_text(summary_obj)
        #   3) summary_user_answer
        final_deep = (
            (deep_text_state or "").strip()
            or self._build_deep_text(summary_obj).strip()
            or summary_user_answer.strip()
            or last_ai_text.strip()
        )

        # DEV:
        #   1) dev_text_state (del grafo: traza markdown completa)
        #   2) last_ai_text
        #   3) deep_text (como fallback)
        final_dev = (
            (dev_text_state or "").strip()
            or last_ai_text.strip()
            or final_deep
        )

        # 9) Construir vistas
        dev_view = AgentView(
            final_answer=final_dev,
            summary=summary_obj,
            tool_runs=tool_runs,
            raw_state=out_state,  # vista completa
        )

        deep_view = AgentView(
            final_answer=final_deep,
            summary=summary_obj,
            tool_runs=tool_runs,
            raw_state={},  # light view
        )

        user_view = AgentView(
            final_answer=final_user,
            summary=summary_obj,
            tool_runs=tool_runs,
            raw_state={},  # usuario nunca necesita estado crudo
        )

        # 10) Actualizar memoria de largo/corto plazo
        try:
            write_memory(
                session_id=session_id,
                user_prompt=prompt_text,
                user_out=final_user,
                user_id=user_id,
                memory_cfg=self.memory_cfg,
            )
        except Exception as e:
            # No queremos que un fallo en memoria rompa el turno
            print(f"[Agent] ⚠️ Error escribiendo memoria: {e!r}")

        # 11) Empaquetar en AgentOutput y devolver como dict puro
        output = AgentOutput(
            dev_out=dev_view,
            deep_out=deep_view,
            user_out=user_view,
        )
        return output.to_dict()

    # ------------------------------------------------------------------
    # Extras útiles para debugging
    # ------------------------------------------------------------------
    @property
    def last_state(self) -> Optional[Dict[str, Any]]:
        """Devuelve el último estado crudo del grafo (solo lectura)."""
        return self._state

    def reset_session(self) -> None:
        """Resetea el estado interno de conversación (no borra memoria global)."""
        self._state = None
