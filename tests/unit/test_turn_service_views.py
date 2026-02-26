from langchain_core.messages import AIMessage

from agnostic_agent.app.turn_service import TurnService
from agnostic_agent.knowledge.types import KnowledgeBase


class _FakeGraph:
    def invoke(self, state):
        return {
            "messages": [
                AIMessage(
                    content="### VALIDATOR\n- all_covered: False",
                    additional_kwargs={"pipeline_internal": True, "node": "validator"},
                ),
                AIMessage(content="Respuesta visible para usuario"),
            ],
            "tool_runs": [],
            "summary": {},
            "pipeline_summary": {},
            "user_out": "",
            "deep_out": "",
            "dev_out": "",
        }


def test_turn_service_prefers_visible_ai_over_pipeline_internal(monkeypatch):
    monkeypatch.setattr("agnostic_agent.app.turn_service.read_memory", lambda session_id: {})
    monkeypatch.setattr(
        "agnostic_agent.app.turn_service.write_memory",
        lambda **kwargs: None,
    )

    svc = TurnService(
        graph_app=_FakeGraph(),
        knowledge_bases=[
            KnowledgeBase(name="kb1", kind="sqlite-vec", config={"path": "dummy.db"})
        ],
        memory_cfg={},
        context_tables=[],
        context_cfg={},
    )

    out = svc.run_turn({"user_prompt": "hola"})
    assert out["user_out"]["final_answer"] == "Respuesta visible para usuario"


def test_turn_service_pipeline_v2_viewmodels(monkeypatch):
    monkeypatch.setattr("agnostic_agent.app.turn_service.read_memory", lambda session_id: {})
    monkeypatch.setattr(
        "agnostic_agent.app.turn_service.write_memory",
        lambda **kwargs: None,
    )
    monkeypatch.setenv("AGNOSTIC_PIPELINE_V2", "true")

    svc = TurnService(
        graph_app=_FakeGraph(),
        knowledge_bases=[
            KnowledgeBase(name="kb1", kind="sqlite-vec", config={"path": "dummy.db"})
        ],
        memory_cfg={},
        context_tables=[],
        context_cfg={},
    )

    out = svc.run_turn({"user_prompt": "hola"})
    assert "## Deep Summary" in out["deep_out"]["final_answer"]
    assert "## Dev Summary" in out["dev_out"]["final_answer"]
    assert "### Solicitud" in out["user_out"]["final_answer"]
