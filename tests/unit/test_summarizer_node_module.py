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
        build_user_answer_from_runs=_build_user_answer_from_runs,
        is_technical_answer=_is_technical_answer,
        find_last_assistant_real=_find_last_assistant_real,
        extract_tool_calls=_extract_tool_calls,
        coerce_content_str=_coerce_content_str,
        strip_think=_strip_think,
    )

    assert out["user_out"].strip()
    assert out["summary"]["final_answer"].strip()
