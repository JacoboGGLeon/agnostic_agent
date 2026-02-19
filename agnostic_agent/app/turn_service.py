from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage

from agnostic_agent.core.models.io_models import (
    AgentInput,
    AgentOutput,
    AgentView,
    AgentSummary,
    ToolRun,
)
from agnostic_agent.logic import State
from agnostic_agent.memory import read_memory, write_memory
from agnostic_agent.knowledge import select_knowledge_bases, KnowledgeBase
from agnostic_agent.app.errors import TurnExecutionError

class TurnService:
    """
    Service responsible for executing a single turn of the agent conversation.
    Decouples the execution logic from the configuration/container (Agent class).
    """

    def __init__(
        self,
        graph_app: Any,
        knowledge_bases: List[KnowledgeBase],
        memory_cfg: Dict[str, Any],
        context_tables: List[str],
        context_cfg: Dict[str, Any],
        setup_path: Optional[str] = None,
        setup_config: Optional[Dict[str, Any]] = None,
    ):
        self.graph_app = graph_app
        self.knowledge_bases = knowledge_bases
        self.memory_cfg = memory_cfg
        self.context_tables = context_tables
        self.context_cfg = context_cfg
        self.setup_path = setup_path
        self.setup_config = setup_config or {}
        
        # Internal state for multi-turn conversation
        self._state: Optional[Dict[str, Any]] = None

    @property
    def last_state(self) -> Optional[Dict[str, Any]]:
        return self._state

    def reset_session(self) -> None:
        self._state = None

    def _coerce_input(
        self,
        user_input: Union[str, Dict[str, Any], AgentInput],
    ) -> AgentInput:
        if isinstance(user_input, AgentInput):
            return user_input
        if isinstance(user_input, dict):
            return AgentInput(**user_input)
        return AgentInput(user_prompt=str(user_input))

    def _clean_prev_messages(self) -> List[AnyMessage]:
        msgs: List[AnyMessage] = []
        if self._state is None:
            return msgs

        for m in self._state.get("messages", []):
            if isinstance(m, ToolMessage):
                continue
            msgs.append(m)
        return msgs

    def _build_deep_text(self, summary_obj: Optional[AgentSummary]) -> str:
        if summary_obj is None:
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

    def run_turn(
        self,
        user_input: Union[str, Dict[str, Any], AgentInput],
    ) -> Dict[str, Any]:
        """
        Executes a conversation turn.
        """
        try:
            agent_in = self._coerce_input(user_input)

            prompt_text = (
                getattr(agent_in, "user_prompt", None)
                or getattr(agent_in, "user_text", None)
                or ""
            )

            # Metadata resolution
            session_id = agent_in.session_id or "default"
            user_id = None
            if agent_in.metadata:
                user_id = agent_in.metadata.get("user_id")

            # Knowledge selection
            knowledge_names = agent_in.knowledge_names
            if not knowledge_names:
                knowledge_names = [kb.name for kb in self.knowledge_bases]

            knowledge_selected = select_knowledge_bases(knowledge_names, self.knowledge_bases)

            # Memory read
            memory_context = read_memory(session_id=session_id)

            # State construction
            prev_messages = self._clean_prev_messages()
            state_in: State = {
                "messages": prev_messages + [HumanMessage(content=prompt_text)],
                "analyzer": None,
                "planner_trajs": [],
                "executor_steps": [],
                "tool_runs": [],
                "summary": None,
                "pipeline_summary": None,
                "user_prompt": prompt_text,
                "session_id": session_id,
                "user_id": user_id,
                "setup_path": self.setup_path or "",
                "setup_config": self.setup_config,
                "knowledge_names": knowledge_names,
                "kb_all": [kb.__dict__ for kb in self.knowledge_bases],
                "knowledge_selected": [kb.__dict__ for kb in knowledge_selected],
                "memory_context": memory_context,
                "context_tables": list(self.context_tables),
                "context_cfg": self.context_cfg,
            }

            # Invoke Graph
            out_state: State = self.graph_app.invoke(state_in)
            self._state = out_state

            # Extract Output
            ai_messages = [
                m for m in out_state.get("messages", []) if isinstance(m, AIMessage)
            ]
            last_ai = ai_messages[-1] if ai_messages else None
            last_ai_text = last_ai.content if last_ai is not None else ""

            dev_text_state = out_state.get("dev_out")
            deep_text_state = out_state.get("deep_out")
            user_text_state = out_state.get("user_out")

            raw_summary: Dict[str, Any] = (
                out_state.get("pipeline_summary")
                or out_state.get("summary")
                or {}
            )
            
            summary_obj = AgentSummary(**raw_summary) if raw_summary else None
            summary_user_answer = summary_obj.final_answer or "" if summary_obj else ""

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

            # Construct Views
            final_user = (
                (user_text_state or "").strip()
                or summary_user_answer.strip()
                or last_ai_text.strip()
            )

            final_deep = (
                (deep_text_state or "").strip()
                or self._build_deep_text(summary_obj).strip()
                or summary_user_answer.strip()
                or last_ai_text.strip()
            )

            final_dev = (
                (dev_text_state or "").strip()
                or last_ai_text.strip()
                or final_deep
            )

            dev_view = AgentView(
                final_answer=final_dev,
                summary=summary_obj,
                tool_runs=tool_runs,
                raw_state=out_state,
            )

            deep_view = AgentView(
                final_answer=final_deep,
                summary=summary_obj,
                tool_runs=tool_runs,
                raw_state={},
            )

            user_view = AgentView(
                final_answer=final_user,
                summary=summary_obj,
                tool_runs=tool_runs,
                raw_state={},
            )

            # Memory Write
            try:
                write_memory(
                    session_id=session_id,
                    user_prompt=prompt_text,
                    user_out=final_user,
                    user_id=user_id,
                    memory_cfg=self.memory_cfg,
                )
            except Exception as e:
                # Log usage but don't fail turn
                print(f"[TurnService] ⚠️ Error writing memory: {e!r}")

            output = AgentOutput(
                dev_out=dev_view,
                deep_out=deep_view,
                user_out=user_view,
            )
            return output.to_dict()

        except AgnosticAgentError as e:
            # Re-raise known errors
            raise e
        except Exception as e:
            # Wrap unexpected errors
            import traceback
            tb = traceback.format_exc()
            raise TurnExecutionError(
                message=f"Unexpected error during turn execution: {str(e)}", 
                step="run_turn", 
                details={"original_error": str(e), "traceback": tb}
            ) from e
