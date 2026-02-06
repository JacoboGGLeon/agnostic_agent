from __future__ import annotations

"""
Clase principal (Agent) del Agnostic Deep Agent 2026.

Orquesta:
- Capabilities (Planner config, modelos).
- Tools (herramientas).
- Context (KnowledgeBases, memoria).
- Logic (grafo LangGraph).

Provee la API unificada:
    agent.run_turn(user_input) → AgentOutput
"""

import os
import uuid
import yaml
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage, AnyMessage

from .communication import (
    BaseAgentInput,
    AgentInput,
    AgentOutput,
    AgentView,
)

from .capabilities import (
    PlannerConfig,
    VllmServers,
    VllmEndpoints,
    prepare_qwen_models,
    start_qwen_vllm_servers,
    build_planner_llm,
)

from .tools import get_default_tools, get_tools_by_names
from .context import KnowledgeBase, get_default_context, get_kb_by_names
from .logic import load_logic, LogicConfig, State


class Agent:
    """
    Agente Agnóstico unificado.

    Uso típico:
        agent = Agent(setup_path="setup.yaml")
        await agent.ainit_resources()  # (opcional, si hay async setup)
        out = agent.run_turn("Hola")
    """

    def __init__(
        self,
        # Configuración
        setup_path: str = "setup.yaml",
        setup_dict: Optional[Dict[str, Any]] = None,
        
        # Inyección manual (override)
        planner_config: Optional[PlannerConfig] = None,
        logic_config: Optional[LogicConfig] = None,
        
        # Estado inicial manual
        initial_memory: Optional[Dict[str, Any]] = None,
        
        # Flags
        auto_start_servers: bool = True,
        verbose: bool = False,
    ):
        """
        Inicializa el agente pero NO arranca modelos pesados automáticamente
        en el constructor, para no bloquear. Llama a `start()` o `ainit_resources()`
        después, o confía en lazy-loading si la lógica lo soporta.
        
        (En esta versión v0.2, sí preparamos configs básicas aquí).
        """
        self.verbose = verbose
        self._memory = initial_memory or {}
        self._session_id = str(uuid.uuid4())

        # 1. Cargar configuración (YAML o Dict)
        self.setup_cfg: Dict[str, Any] = {}
        if setup_dict:
            self.setup_cfg = setup_dict
        elif setup_path and os.path.exists(setup_path):
            try:
                with open(setup_path, "r", encoding="utf-8") as f:
                    self.setup_cfg = yaml.safe_load(f) or {}
            except Exception as e:
                if verbose:
                    print(f"[Agent] Warning: could not load {setup_path}: {e}")
        
        # 2. Configurar Planner y Modelos
        #    Prioridad: argumento __init__ > setup.yaml > defaults
        if planner_config:
            self.planner_config = planner_config
        else:
            # Intentar leer de setup.yaml
            p_cfg = self.setup_cfg.get("planner") or {}
            self.planner_config = PlannerConfig(**p_cfg)

        # 3. KBs disponibles (contexto global)
        #    Se cargan desde setup.yaml (sección knowledge_bases)
        self.available_kbs = get_default_context(self.setup_cfg)

        # 4. Tools disponibles (globales)
        #    Se asume que TOOL_REGISTRY tiene todo. Podríamos filtrar si setup.yaml lo pide.
        #    Aquí cargamos TODAS por defecto para que el Planner decida.
        self.available_tools = get_default_tools()

        # 5. Lógica (Grafo)
        self.logic_config = logic_config or LogicConfig()  # defaults
        
        # Estado interno del grafo (persistencia en memoria ram por ahora)
        self.graph_app = None  # Se compila al iniciar
        self._state: State = {} # Último estado conocido

        if auto_start_servers:
            # En un entorno real, esto debería ser async o explícito
            self._bootstrap_sync()


    def _bootstrap_sync(self):
        """
        Arranca servidores y compila el grafo de forma síncrona.
        (Cuidado: descarga modelos si no existen).
        """
        if self.verbose:
            print("[Agent] Bootstrapping resources...")

        # 1. Preparar modelos (download si hace falta)
        #    Esto lee de setup.yaml -> models -> qwen -> ...
        model_paths = prepare_qwen_models(self.setup_cfg)
        
        # 2. Arrancar vLLM (si config lo pide y no están activos)
        #    start_qwen_vllm_servers es inteligente (comprueba puertos).
        endpoints: VllmEndpoints = start_qwen_vllm_servers(self.setup_cfg, model_paths)
        
        # Actualizamos la config del planner con los endpoints reales
        # (por si cambiaron puertos o es la primera vez)
        if endpoints.planner_url:
            self.planner_config.vllm_base_url = endpoints.planner_url
            self.planner_config.model_name = endpoints.planner_model_name
            # Asegurar apiKey dummy si vLLM no tiene auth
            if not self.planner_config.vllm_api_key:
                self.planner_config.vllm_api_key = "EMPTY"

        # 3. Construir LLM object
        self.planner_llm = build_planner_llm(self.planner_config)

        # 4. Compilar Grafo (Logic)
        self.graph_app = load_logic(
            planner_llm=self.planner_llm,
            tools=self.available_tools,
            planner_config=self.planner_config,
            logic_config=self.logic_config,
        )

        if self.verbose:
            print("[Agent] Ready.")


    def run_turn(
        self,
        user_input: Union[str, Dict[str, Any], BaseAgentInput],
    ) -> Dict[str, Any]:
        """
        Ejecuta un turno conversación.

        Admite:
        - str: "Hola agente"
        - dict: {"prompt": "Hola", ...}
        - AgentInput: objeto tipado
        
        Devuelve dict compatible con AgentOutput (dev_out, deep_out, user_out).
        """
        # 1. Normalizar entrada
        if isinstance(user_input, str):
            inp = AgentInput(prompt=user_input)
        elif isinstance(user_input, dict):
            inp = AgentInput(**user_input)
        elif isinstance(user_input, BaseAgentInput):  # (BaseAgentInput es AgentInput)
            inp = user_input
        else:
            raise ValueError(f"Input type not supported: {type(user_input)}")

        if not inp.prompt:
            # Caso borde: prompt vacío
            return AgentOutput(
                dev_out=AgentView(content=""),
                deep_out=AgentView(content=""),
                user_out=AgentView(content="Error: prompt vacío."),
            ).to_dict()

        # 2. Resolver Session / KBs
        #    Si inp.session_id viene, lo usamos. Si no, self._session_id.
        session_id = inp.session_id or self._session_id
        
        #    Si inp.context_files viene, creamos KBs al vuelo y las sumamos
        #    a las available_kbs para este turno.
        #    (Nota: la lógica actual filter por nombre, así que habría que darles nombre).
        #    Por simplicidad en v0.2:
        #      - Tomamos todas las available_kbs
        #      - + KBs on-the-fly si input lo pide (no implementado fully aquí, pero preparado).
        
        current_kb_names = [kb.name for kb in self.available_kbs]
        
        # 3. Leer memoria (simple dict)
        #    En el futuro: MemoryManager.load(session_id)
        mem_ctx = self._memory.get(session_id, {})

        # 4. Preparar estado inicial para LangGraph
        #    Si es el primer mensaje de la sesión, messages inicia vacío o con historial cargado.
        #    LangGraph con checkpointer maneja esto, pero aquí lo hacemos manual/in-memory para demo.
        
        # Recuperar historial previo si existiera en self._state (muy naif)
        # Lo ideal es tener un `history` real. Aquí asumimos stateless entre turnos 
        # SALVO que inyectemos `messages` previos. Para simplificar la demo:
        # Pasamos el historial que venga en `state` si persistimos `self._state`,
        # pero `self._state` se machaca en cada run_turn.
        
        # FIX v0.2: Usar hilo conversacional real requiere persistencia.
        # Aquí reconstruimos `messages` con lo que tengamos + el nuevo HumanMessage.
        
        prev_messages = self._state.get("messages", []) if self._state else []
        new_messages = list(prev_messages) + [HumanMessage(content=inp.prompt)]

        state_in: State = {
            "messages": new_messages,
            "user_prompt": inp.prompt,
            "session_id": session_id,
            "kb_names": current_kb_names,
            "memory_context": mem_ctx,
            # Limpiamos trazas anteriores para que no se mezclen en la visualización
            "planner_trajs": [],
            "executor_steps": [],
            "tool_runs": [],
            "analyzer": {},
            "summary": {},
            "pipeline_summary": {},
            "validator": {},
            "dev_out": "",
            "deep_out": "",
            "user_out": "",
        }

        # 5. Invocar Grafo
        if not self.graph_app:
            self._bootstrap_sync()
        
        # invoke devuelve el estado final
        out_state: State = self.graph_app.invoke(state_in)
        
        # 6. Guardar estado (para siguiente turno en memoria volátil)
        self._state = out_state
        
        # 7. Actualizar memoria (escritura simple)
        #    Si quisiéramos recordar algo, el Summarizer o un nodo MemoryWriter
        #    debería haber actualizado `memory_context`.
        #    Aquí simulamos que no cambia automáticamente salvo que lo hagamos explícito.
        
        # 8. Empaquetar Output
        dev_view = AgentView(
            content=out_state.get("dev_out") or "",
            metadata={"state_keys": list(out_state.keys())}
        )
        deep_view = AgentView(
            content=out_state.get("deep_out") or "",
            metadata={"summary": out_state.get("summary")}
        )
        user_view = AgentView(
            content=out_state.get("user_out") or "",
            metadata={"validator": out_state.get("validator")}
        )
        
        output = AgentOutput(
            dev_out=dev_view,
            deep_out=deep_view,
            user_out=user_view,
        )
        
        return output.to_dict()
