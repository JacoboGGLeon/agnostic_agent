import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agnostic_agent.graph.analyzer_node import execute_analyzer_node


class _StubLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, _messages):
        return AIMessage(content=self._content)


def test_execute_analyzer_node_splits_multi_json_prompt():
    llm_payload = json.dumps(
        {
            "subqueries": [
                "Realiza la conciliacion de los siguientes creditos"
            ],
            "logic_form": "q1",
            "selected_skills": ["contabilidad_instantanea"],
        },
        ensure_ascii=False,
    )
    state = {
        "messages": [
            HumanMessage(
                content=(
                    'Realiza la conciliación: {"credito_id":"LOC-1","saldo_total":10}, '
                    '{"credito_id":"LOC-2","saldo_total":20}'
                )
            )
        ]
    }

    out = execute_analyzer_node(
        state,
        tools=[],
        cfg=None,
        planner_llm=_StubLLM(llm_payload),
        skill_registry=None,
        ai_message_type=AIMessage,
        human_message_type=HumanMessage,
        system_message_type=SystemMessage,
        coerce_content_str=lambda x: x if isinstance(x, str) else str(x),
        sanitize_subquery_text=lambda s: str(s).strip(),
        extract_top_level_json_objects=lambda _t: [
            '{"credito_id":"LOC-1","saldo_total":10}',
            '{"credito_id":"LOC-2","saldo_total":20}',
        ],
        is_placeholder_subquery=lambda _s: False,
    )

    assert "analyzer" in out
    assert len(out["analyzer"]["subqueries"]) == 2
    assert out["_active_skills_internal"] == ["contabilidad_instantanea"]
