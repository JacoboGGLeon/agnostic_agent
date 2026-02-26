from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from agnostic_agent.app.errors import AgnosticAgentError, TurnExecutionError
from agnostic_agent.core.models.io_models import (
    AgentInput,
    AgentOutput,
    AgentSummary,
    AgentView,
    ToolRun,
)
from agnostic_agent.core.pipeline_v2 import (
    build_pipeline_output_v2,
    render_deep_text,
    render_dev_text,
    render_user_text,
)
from agnostic_agent.knowledge import KnowledgeBase, select_knowledge_bases
from agnostic_agent.logic import State
from agnostic_agent.memory import read_memory, write_memory

logger = logging.getLogger(__name__)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            import json

            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


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
            if isinstance(m, AIMessage):
                tc = getattr(m, "tool_calls", None)
                if isinstance(tc, list) and tc:
                    continue
                addkw = getattr(m, "additional_kwargs", {}) or {}
                tc2 = addkw.get("tool_calls") if isinstance(addkw, dict) else None
                if isinstance(tc2, list) and tc2:
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

    def _resolve_prompt_text(self, agent_in: AgentInput) -> str:
        return (
            getattr(agent_in, "user_prompt", None)
            or getattr(agent_in, "user_text", None)
            or ""
        )

    def _resolve_metadata(self, agent_in: AgentInput) -> Dict[str, Any]:
        metadata = agent_in.metadata or {}
        session_id = agent_in.session_id or "default"
        user_id = metadata.get("user_id")
        forced_skill = metadata.get("forced_skill")
        skills_allowlist = None

        raw_history_enabled = metadata.get("conversation_history_enabled", True)
        if isinstance(raw_history_enabled, str):
            conversation_history_enabled = raw_history_enabled.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        else:
            conversation_history_enabled = bool(raw_history_enabled)

        raw_allow = metadata.get("skills_allowlist")
        if isinstance(raw_allow, str) and raw_allow.strip():
            skills_allowlist = [s.strip() for s in raw_allow.split(",") if s.strip()]
        elif isinstance(raw_allow, list):
            skills_allowlist = [str(s).strip() for s in raw_allow if str(s).strip()]

        raw_pipeline_v2 = metadata.get("pipeline_v2")
        if isinstance(raw_pipeline_v2, str):
            pipeline_v2_enabled = raw_pipeline_v2.strip().lower() in {"1", "true", "yes", "y", "on"}
        elif raw_pipeline_v2 is None:
            pipeline_v2_enabled = False
        else:
            pipeline_v2_enabled = bool(raw_pipeline_v2)

        return {
            "session_id": session_id,
            "user_id": user_id,
            "forced_skill": forced_skill,
            "skills_allowlist": skills_allowlist,
            "conversation_history_enabled": conversation_history_enabled,
            "pipeline_v2_enabled": pipeline_v2_enabled,
        }

    def _use_pipeline_v2(self, meta: Dict[str, Any]) -> bool:
        if bool(meta.get("pipeline_v2_enabled")):
            return True
        import os

        env_val = os.getenv("AGNOSTIC_PIPELINE_V2", "")
        return env_val.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _resolve_knowledge_names(self, agent_in: AgentInput) -> List[str]:
        if agent_in.knowledge_names:
            return list(agent_in.knowledge_names)
        return [kb.name for kb in self.knowledge_bases]

    def _build_state_in(
        self,
        *,
        prompt_text: str,
        meta: Dict[str, Any],
        knowledge_names: List[str],
        knowledge_selected: List[KnowledgeBase],
        memory_context: Dict[str, Any],
    ) -> State:
        prev_messages = self._clean_prev_messages() if meta["conversation_history_enabled"] else []

        return {
            "messages": prev_messages + [HumanMessage(content=prompt_text)],
            "analyzer": None,
            "planner_trajs": [],
            "executor_steps": [],
            "tool_runs": [],
            "summary": None,
            "pipeline_summary": None,
            "user_prompt": prompt_text,
            "session_id": meta["session_id"],
            "user_id": meta["user_id"],
            # Skill selection controls:
            # - forced_skill: legacy single-skill selector (UI). Semantically: allowlist of one.
            # - skills_allowlist: preferred multi-skill allowlist.
            "forced_skill": meta["forced_skill"],
            "skills_allowlist": meta["skills_allowlist"],
            "setup_path": self.setup_path or "",
            "setup_config": self.setup_config,
            "knowledge_names": knowledge_names,
            "kb_all": [kb.__dict__ for kb in self.knowledge_bases],
            "knowledge_selected": [kb.__dict__ for kb in knowledge_selected],
            "memory_context": memory_context,
            "context_tables": list(self.context_tables),
            "context_cfg": self.context_cfg,
        }

    def _extract_last_ai_text(self, out_state: State) -> str:
        ai_messages = [m for m in out_state.get("messages", []) if isinstance(m, AIMessage)]
        visible_ai_messages = []
        for msg in ai_messages:
            addkw = getattr(msg, "additional_kwargs", {}) or {}
            if isinstance(addkw, dict) and addkw.get("pipeline_internal"):
                continue
            visible_ai_messages.append(msg)

        last_ai = (
            visible_ai_messages[-1]
            if visible_ai_messages
            else (ai_messages[-1] if ai_messages else None)
        )
        return _safe_text(last_ai.content if last_ai is not None else "")

    def _build_summary_obj(self, out_state: State) -> Optional[AgentSummary]:
        raw_summary: Dict[str, Any] = (
            out_state.get("pipeline_summary")
            or out_state.get("summary")
            or {}
        )
        return AgentSummary(**raw_summary) if raw_summary else None

    def _build_tool_runs(self, out_state: State) -> List[ToolRun]:
        raw_runs = out_state.get("tool_runs", []) or []
        tool_runs: List[ToolRun] = []
        for run in raw_runs:
            tool_runs.append(
                ToolRun(
                    id=str(run.get("id", "")),
                    name=str(run.get("name", "")),
                    args=run.get("args", {}),
                    output=run.get("output"),
                )
            )
        return tool_runs

    def _build_views(
        self,
        *,
        out_state: State,
        summary_obj: Optional[AgentSummary],
        tool_runs: List[ToolRun],
        last_ai_text: str,
    ) -> Dict[str, AgentView]:
        dev_text_state = _safe_text(out_state.get("dev_out"))
        deep_text_state = _safe_text(out_state.get("deep_out"))
        user_text_state = _safe_text(out_state.get("user_out"))
        summary_user_answer = _safe_text(summary_obj.final_answer or "") if summary_obj else ""

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
        final_dev = (dev_text_state or "").strip() or last_ai_text.strip() or final_deep

        return {
            "dev": AgentView(
                final_answer=final_dev,
                summary=summary_obj,
                tool_runs=tool_runs,
                raw_state=out_state,
            ),
            "deep": AgentView(
                final_answer=final_deep,
                summary=summary_obj,
                tool_runs=tool_runs,
                raw_state={},
            ),
            "user": AgentView(
                final_answer=final_user,
                summary=summary_obj,
                tool_runs=tool_runs,
                raw_state={},
            ),
        }

    def _build_views_v2(
        self,
        *,
        prompt_text: str,
        out_state: State,
        summary_obj: Optional[AgentSummary],
        tool_runs: List[ToolRun],
        fallback_final_user: str,
    ) -> Dict[str, AgentView]:
        v2_output = build_pipeline_output_v2(
            prompt_text=prompt_text,
            out_state=out_state,
            summary_obj=summary_obj,
            tool_runs=tool_runs,
            fallback_final_user=fallback_final_user,
        )

        user_text = render_user_text(v2_output.user_out)
        deep_text = render_deep_text(v2_output.deep_out)
        dev_text = render_dev_text(v2_output.dev_out)

        raw_bundle = {
            "pipeline_v2": v2_output.model_dump(),
            "state": out_state,
        }

        return {
            "dev": AgentView(
                final_answer=dev_text,
                summary=summary_obj,
                tool_runs=tool_runs,
                raw_state=raw_bundle,
            ),
            "deep": AgentView(
                final_answer=deep_text,
                summary=summary_obj,
                tool_runs=tool_runs,
                raw_state={},
            ),
            "user": AgentView(
                final_answer=user_text,
                summary=summary_obj,
                tool_runs=tool_runs,
                raw_state={},
            ),
        }

    def _persist_memory(
        self,
        *,
        session_id: str,
        prompt_text: str,
        final_user: str,
        user_id: Optional[str],
    ) -> None:
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
            logger.warning("turn_service memory write failed: %r", e)

    def run_turn(
        self,
        user_input: Union[str, Dict[str, Any], AgentInput],
    ) -> Dict[str, Any]:
        """
        Executes a conversation turn.
        """
        try:
            agent_in = self._coerce_input(user_input)
            prompt_text = self._resolve_prompt_text(agent_in)
            meta = self._resolve_metadata(agent_in)
            knowledge_names = self._resolve_knowledge_names(agent_in)
            knowledge_selected = select_knowledge_bases(knowledge_names, self.knowledge_bases)
            memory_context = read_memory(session_id=meta["session_id"])

            state_in = self._build_state_in(
                prompt_text=prompt_text,
                meta=meta,
                knowledge_names=knowledge_names,
                knowledge_selected=knowledge_selected,
                memory_context=memory_context,
            )

            out_state: State = self.graph_app.invoke(state_in)
            self._state = out_state

            last_ai_text = self._extract_last_ai_text(out_state)
            summary_obj = self._build_summary_obj(out_state)
            tool_runs = self._build_tool_runs(out_state)
            views = self._build_views(
                out_state=out_state,
                summary_obj=summary_obj,
                tool_runs=tool_runs,
                last_ai_text=last_ai_text,
            )
            if self._use_pipeline_v2(meta):
                views = self._build_views_v2(
                    prompt_text=prompt_text,
                    out_state=out_state,
                    summary_obj=summary_obj,
                    tool_runs=tool_runs,
                    fallback_final_user=views["user"].final_answer,
                )

            self._persist_memory(
                session_id=meta["session_id"],
                prompt_text=prompt_text,
                final_user=views["user"].final_answer,
                user_id=meta["user_id"],
            )

            output = AgentOutput(
                dev_out=views["dev"],
                deep_out=views["deep"],
                user_out=views["user"],
            )
            return output.to_dict()

        except AgnosticAgentError as e:
            raise e
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            raise TurnExecutionError(
                message=f"Unexpected error during turn execution: {str(e)}",
                step="run_turn",
                details={"original_error": str(e), "traceback": tb},
            ) from e
