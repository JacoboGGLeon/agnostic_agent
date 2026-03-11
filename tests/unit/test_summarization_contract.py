from agnostic_agent.graph.summarization import (
    build_agnostic_user_answer,
    looks_like_technical_answer,
)


def test_looks_like_technical_answer_detects_tool_log_shape():
    text = (
        "Se ejecutaron 3 tools.\n"
        "1. reconcile_credit_accounting -> status=ok\n"
        "2. get_rate -> resultado=dict\n"
        "3. query_db -> resultado=dict\n"
    )
    assert looks_like_technical_answer(text) is True


def test_build_agnostic_user_answer_returns_user_facing_markdown():
    runs = [
        {"name": "tool_a", "args": {"record_id": "R-1"}, "output": {"ok": True, "status": "done"}},
        {"name": "tool_b", "args": {}, "output": {"error": "timeout"}},
    ]
    out = build_agnostic_user_answer("haz algo", runs)
    assert out.strip()
    assert "tool_call_id" not in out
    assert "Detecte 1 ejecuciones con error" in out
    assert "Deep/Dev" not in out
