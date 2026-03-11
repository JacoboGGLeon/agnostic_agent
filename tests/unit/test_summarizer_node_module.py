from langchain_core.messages import AIMessage, HumanMessage

from agnostic_agent.graph.summarizer_node import execute_summarizer_node


class _StubPlannerLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, _messages):
        return AIMessage(content=self._content)


def _default_json(value):
    return repr(value)


def _summarize_tool_runs(_user_prompt, _runs):
    return "tool-summary"


def _summarize_tool_runs_compact(runs):
    return f"Se ejecutaron {len(runs)} tools."


def _build_user_answer_from_runs(_user_prompt, runs):
    return f"## Resultado\nSe procesaron {len(runs)} ejecuciones."


def _build_response_bundle(_user_prompt, runs, _analyzer_subqueries=None):
    return {
        "kind": "tool_evidence",
        "items": [{"label": "Solicitud", "message": f"Se procesaron {len(runs)} ejecuciones."}],
        "errors": 0,
        "findings": [],
    }


def _render_response_bundle(bundle, level="user"):
    msg = bundle["items"][0]["message"]
    if level == "user":
        return msg
    return f"{level}: {msg}"


def _is_technical_answer(_text: str) -> bool:
    return False


def _resolve_effective_skills(_state, _registry):
    return ["contabilidad_instantanea"]


def _find_last_assistant_real(messages):
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _extract_tool_calls(_msg):
    return []


def _coerce_content_str(value):
    return value if isinstance(value, str) else str(value)


def _strip_think(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "")


def test_execute_summarizer_node_preserves_llm_clean_output_when_no_tools():
    state = {
        "messages": [HumanMessage(content="hola")],
        "tool_runs": [],
        "llm_clean_out": "respuesta directa",
        "analyzer": {"subqueries": []},
        "planner_trajs": [],
        "executor_steps": [],
    }

    out = execute_summarizer_node(
        state,
        skill_registry=None,
        tools=[],
        cfg=None,
        planner_llm=_StubPlannerLLM("unused"),
        resolve_effective_skills=_resolve_effective_skills,
        json_default=_default_json,
        summarize_tool_runs=_summarize_tool_runs,
        summarize_tool_runs_compact=_summarize_tool_runs_compact,
        build_response_bundle=_build_response_bundle,
        render_response_bundle=_render_response_bundle,
        build_user_answer_from_runs=_build_user_answer_from_runs,
        is_technical_answer=_is_technical_answer,
        find_last_assistant_real=_find_last_assistant_real,
        extract_tool_calls=_extract_tool_calls,
        coerce_content_str=_coerce_content_str,
        strip_think=_strip_think,
    )

    assert out["user_out"] == "respuesta directa"
    assert out["summary"]["final_answer"] == "respuesta directa"


def test_execute_summarizer_node_never_returns_empty_user_out_with_tools():
    runs = [
        {
            "name": "reconcile_credit_accounting",
            "args": {"credito_id": "LOC-0004"},
            "output": {"ok": True, "status": "CUADRADO (100% Match)"},
        }
    ]
    state = {
        "messages": [HumanMessage(content="conciliar")],
        "user_prompt": "conciliar",
        "tool_runs": runs,
        "analyzer": {"subqueries": ["q1"]},
        "planner_trajs": [{"subquery": "q1", "description": "step 1"}],
        "executor_steps": [{"tool_call_id": "c1", "tool_name": "reconcile_credit_accounting", "args": {}}],
    }

    out = execute_summarizer_node(
        state,
        skill_registry=None,
        tools=[],
        cfg=None,
        planner_llm=_StubPlannerLLM(""),
        resolve_effective_skills=_resolve_effective_skills,
        json_default=_default_json,
        summarize_tool_runs=_summarize_tool_runs,
        summarize_tool_runs_compact=_summarize_tool_runs_compact,
        build_response_bundle=_build_response_bundle,
        render_response_bundle=_render_response_bundle,
        build_user_answer_from_runs=_build_user_answer_from_runs,
        is_technical_answer=_is_technical_answer,
        find_last_assistant_real=_find_last_assistant_real,
        extract_tool_calls=_extract_tool_calls,
        coerce_content_str=_coerce_content_str,
        strip_think=_strip_think,
    )

    assert out["user_out"].strip()
    assert out["summary"]["final_answer"].strip()
    assert "deep" not in out["user_out"].lower()
    assert "RESPUESTA FINAL" in out["deep_out"]


def test_execute_summarizer_node_uses_grounded_llm_for_user_view_when_safe():
    runs = [
        {
            "name": "nl2sql",
            "args": {"user_request": "dame el top 2"},
            "output": {"ok": True, "execution": {"ok": True, "row_count": 2}},
        }
    ]
    state = {
        "messages": [HumanMessage(content="dame el top 2 creditos con saldo mas alto")],
        "user_prompt": "dame el top 2 creditos con saldo mas alto",
        "tool_runs": runs,
        "analyzer": {"subqueries": ["q1"]},
        "planner_trajs": [{"subquery": "q1", "description": "step 1"}],
        "executor_steps": [{"tool_call_id": "c1", "tool_name": "nl2sql", "args": {}}],
    }

    out = execute_summarizer_node(
        state,
        skill_registry=None,
        tools=[],
        cfg=None,
        planner_llm=_StubPlannerLLM("Los dos creditos con saldo mas alto ya quedaron identificados."),
        resolve_effective_skills=_resolve_effective_skills,
        json_default=_default_json,
        summarize_tool_runs=_summarize_tool_runs,
        summarize_tool_runs_compact=_summarize_tool_runs_compact,
        build_response_bundle=_build_response_bundle,
        render_response_bundle=_render_response_bundle,
        build_user_answer_from_runs=_build_user_answer_from_runs,
        is_technical_answer=_is_technical_answer,
        find_last_assistant_real=_find_last_assistant_real,
        extract_tool_calls=_extract_tool_calls,
        coerce_content_str=_coerce_content_str,
        strip_think=_strip_think,
    )

    assert out["user_out"] == "Los dos creditos con saldo mas alto ya quedaron identificados."
    assert "dev:" in out["dev_out"]
