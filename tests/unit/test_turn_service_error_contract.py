import pytest

from agnostic_agent.app.errors import TurnExecutionError
from agnostic_agent.app.turn_service import TurnService
from agnostic_agent.knowledge.types import KnowledgeBase


class _BrokenGraph:
    def invoke(self, state):
        raise RuntimeError("graph exploded")


def test_turn_service_wraps_unexpected_errors(monkeypatch):
    monkeypatch.setattr("agnostic_agent.app.turn_service.read_memory", lambda session_id: {})
    monkeypatch.setattr("agnostic_agent.app.turn_service.write_memory", lambda **kwargs: None)

    svc = TurnService(
        graph_app=_BrokenGraph(),
        knowledge_bases=[KnowledgeBase(name="kb1", kind="sqlite-vec", config={"path": "dummy.db"})],
        memory_cfg={},
        context_tables=[],
        context_cfg={},
    )

    with pytest.raises(TurnExecutionError) as exc:
        svc.run_turn({"user_prompt": "hola"})
    assert exc.value.code == "TURN_EXECUTION_ERROR"
    assert exc.value.details.get("step") == "run_turn"
    assert "original_error" in exc.value.details
