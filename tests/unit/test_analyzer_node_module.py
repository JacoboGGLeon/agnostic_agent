import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agnostic_agent.graph.analyzer_node import execute_analyzer_node


class _Skill:
    def __init__(self, name, *, world=None):
        self.name = name
        self.world = world or name
        self.description = name
        self.tools = ["nl2sql"]
        self.knowledge = []
        self.intents = ["query_financial_data", "reconcile_credit", "batch_reconcile"]
        self.entities = ["credito_id"]
        self.planner_policy = {}
        self.summarizer_policy = {}
        self.validator_policy = {}
        self.ui = {}
        self.instructions = ""


class _Registry:
    def __init__(self, skills):
        self._skills = {s.name: s for s in skills}

    def get_skill(self, name):
        return self._skills.get(name)

    def get_world(self, name):
        return self._skills.get(name)

    def list_skills(self):
        return list(self._skills.values())


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
    assert out["analyzer"]["selected_skill_world"] == "contabilidad_instantanea"
    assert out["analyzer"]["selection_mode"] == "auto"
    assert len(out["analyzer"]["subquery_intents"]) == 2
    assert len(out["analyzer"]["entities_by_subquery"]) == 2
    assert out["selected_skill_world"] == "contabilidad_instantanea"


def test_execute_analyzer_node_expands_finance_query_into_multi_source_propositions():
    llm_payload = json.dumps(
        {
            "subqueries": [
                "dame información sobre el crédito LOC-0004 de tus bases de datos"
            ],
            "logic_form": "q1",
            "selected_skills": ["contabilidad_automatica"],
            "selected_skill_world": "contabilidad_automatica",
        },
        ensure_ascii=False,
    )
    state = {
        "messages": [HumanMessage(content="dame información sobre el crédito LOC-0004 de tus bases de datos")],
        "forced_skill": "contabilidad_automatica",
        "skills_allowlist": ["contabilidad_automatica"],
    }

    out = execute_analyzer_node(
        state,
        tools=[],
        cfg=None,
        planner_llm=_StubLLM(llm_payload),
        skill_registry=_Registry([_Skill("contabilidad_automatica")]),
        ai_message_type=AIMessage,
        human_message_type=HumanMessage,
        system_message_type=SystemMessage,
        coerce_content_str=lambda x: x if isinstance(x, str) else str(x),
        sanitize_subquery_text=lambda s: str(s).strip(),
        extract_top_level_json_objects=lambda _t: [],
        is_placeholder_subquery=lambda _s: False,
    )

    analyzer = out["analyzer"]
    assert analyzer["selected_skill_world"] == "contabilidad_automatica"
    assert analyzer["propositional_logic"] == "q1 AND q2"
    assert analyzer["source_scope"] == "multi_source"
    assert analyzer["composition_mode"] == "merge"
    assert analyzer["coverage_expectation"] == "composite"
    assert analyzer["decomposition_strategy"] == "finance_cross_source_split"
    assert analyzer["subqueries"] == [
        "snapshot contable del crédito LOC-0004 en contabilidad.db",
        "movimientos del crédito LOC-0004 en transacciones.db",
    ]
    assert analyzer["required_sources_by_subquery"] == [["contabilidad.db"], ["transacciones.db"]]
    assert analyzer["entities_by_subquery"][0]["credito_id"] == "LOC-0004"
    assert analyzer["entities_by_subquery"][1]["credito_id"] == "LOC-0004"


def test_execute_analyzer_node_expands_plain_text_batch_entities_into_atomic_subqueries():
    llm_payload = json.dumps(
        {
            "subqueries": [
                "Realiza la conciliación de los siguientes créditos: LOC-0004 LOC-0005 LOC-0006 LOC-0007 LOC-0008"
            ],
            "logic_form": "q1",
            "selected_skills": ["contabilidad_automatica"],
            "selected_skill_world": "contabilidad_automatica",
        },
        ensure_ascii=False,
    )
    state = {
        "messages": [HumanMessage(content="Realiza la conciliación de los siguientes créditos: LOC-0004 LOC-0005 LOC-0006 LOC-0007 LOC-0008")],
        "forced_skill": "contabilidad_automatica",
        "skills_allowlist": ["contabilidad_automatica"],
    }

    out = execute_analyzer_node(
        state,
        tools=[],
        cfg=None,
        planner_llm=_StubLLM(llm_payload),
        skill_registry=_Registry([_Skill("contabilidad_automatica")]),
        ai_message_type=AIMessage,
        human_message_type=HumanMessage,
        system_message_type=SystemMessage,
        coerce_content_str=lambda x: x if isinstance(x, str) else str(x),
        sanitize_subquery_text=lambda s: str(s).strip(),
        extract_top_level_json_objects=lambda _t: [],
        is_placeholder_subquery=lambda _s: False,
    )

    analyzer = out["analyzer"]
    assert analyzer["propositional_logic"] == "q1 AND q2 AND q3 AND q4 AND q5"
    assert analyzer["decomposition_strategy"] == "batch_entity_split"
    assert analyzer["response_mode"] == "batch_summary"
    assert len(analyzer["subqueries"]) == 5
    assert analyzer["entities_by_subquery"] == [
        {"credito_id": "LOC-0004"},
        {"credito_id": "LOC-0005"},
        {"credito_id": "LOC-0006"},
        {"credito_id": "LOC-0007"},
        {"credito_id": "LOC-0008"},
    ]


def test_execute_analyzer_node_resolves_referential_batch_from_memory_context():
    llm_payload = json.dumps(
        {
            "subqueries": ["concílialos"],
            "logic_form": "q1",
            "selected_skills": ["contabilidad_automatica"],
            "selected_skill_world": "contabilidad_automatica",
        },
        ensure_ascii=False,
    )
    state = {
        "messages": [HumanMessage(content="concílialos")],
        "forced_skill": "contabilidad_automatica",
        "skills_allowlist": ["contabilidad_automatica"],
        "memory_context": {
            "working_memory": {
                "last_listed_entities_by_type": {
                    "credito_id": ["LOC-0004", "LOC-0005", "LOC-0006"]
                },
                "active_entities_by_type": {
                    "credito_id": ["LOC-0004", "LOC-0005", "LOC-0006"]
                },
                "last_operation": "data_lookup",
            }
        },
    }

    out = execute_analyzer_node(
        state,
        tools=[],
        cfg=None,
        planner_llm=_StubLLM(llm_payload),
        skill_registry=_Registry([_Skill("contabilidad_automatica")]),
        ai_message_type=AIMessage,
        human_message_type=HumanMessage,
        system_message_type=SystemMessage,
        coerce_content_str=lambda x: x if isinstance(x, str) else str(x),
        sanitize_subquery_text=lambda s: str(s).strip(),
        extract_top_level_json_objects=lambda _t: [],
        is_placeholder_subquery=lambda _s: False,
    )

    analyzer = out["analyzer"]
    assert analyzer["propositional_logic"] == "q1 AND q2 AND q3"
    assert analyzer["decomposition_strategy"] == "memory_reference_batch_split"
    assert len(analyzer["subqueries"]) == 3
    assert analyzer["entities_by_subquery"] == [
        {"credito_id": "LOC-0004"},
        {"credito_id": "LOC-0005"},
        {"credito_id": "LOC-0006"},
    ]
