import pytest
from langchain_core.messages import AIMessage

from agnostic_agent.app.errors import TurnExecutionError
from agnostic_agent.app.turn_service import TurnService
from agnostic_agent.knowledge.types import KnowledgeBase


class _FakeGraph:
    def invoke(self, state):
        return {
            "messages": [AIMessage(content="Respuesta visible para usuario")],
            "tool_runs": [],
            "summary": {},
            "pipeline_summary": {},
            "user_out": "ok",
            "deep_out": "",
            "dev_out": "",
        }


def test_turn_service_strict_protocol_enforcement_raises_when_srp_invalid(monkeypatch):
    monkeypatch.setattr("agnostic_agent.app.turn_service.read_memory", lambda session_id: {})
    monkeypatch.setattr("agnostic_agent.app.turn_service.write_memory", lambda **kwargs: None)
    monkeypatch.setenv("AGNOSTIC_STRICT_PROTOCOLS", "true")
    monkeypatch.setattr("agnostic_agent.app.turn_service.validate_srp_response", lambda _rsp: (False, ["bad"]))

    svc = TurnService(
        graph_app=_FakeGraph(),
        knowledge_bases=[KnowledgeBase(name="kb1", kind="sqlite-vec", config={"path": "dummy.db"})],
        memory_cfg={},
        context_tables=[],
        context_cfg={},
    )

    with pytest.raises(TurnExecutionError) as exc:
        svc.run_turn({"user_prompt": "hola"})
    assert exc.value.details.get("step") == "protocol_enforcement"
