from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agnostic_agent.graph.planner_node import execute_planner_node


class _Skill:
    def __init__(self, name, tools=None, knowledge=None, world=None, intents=None, planner_policy=None):
        self.name = name
        self.tools = tools or []
        self.knowledge = knowledge or []
        self.description = name
        self.world = world or name
        self.intents = intents or []
        self.entities = []
        self.planner_policy = planner_policy or {}
        self.summarizer_policy = {}
        self.validator_policy = {}
        self.ui = {}


class _Registry:
    def __init__(self, skills):
        self._skills = {s.name: s for s in skills}

    def get_skill(self, name):
        return self._skills.get(name)

    def get_world(self, name):
        return self._skills.get(name)

    def list_skills(self):
        return list(self._skills.values())


class _Tool:
    def __init__(self, name):
        self.name = name
        self.description = "tool"


class _PlannerLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return self._responses.pop(0)


def test_execute_planner_node_deduplicates_calls_across_subqueries():
    llm = _PlannerLLM(
        [
            AIMessage(content="ok", tool_calls=[{"id": "a1", "name": "tool_a", "args": {"x": 1}}]),
            AIMessage(content="ok", tool_calls=[{"id": "a2", "name": "tool_a", "args": {"x": 1}}]),
        ]
    )
    state = {
        "messages": [HumanMessage(content="haz plan")],
        "analyzer": {"subqueries": ["q1", "q2"]},
    }
    out = execute_planner_node(
        state,
        tools=[_Tool("tool_a")],
        cfg=type("Cfg", (), {"enable_thinking": False, "max_retries": 0})(),
        planner_llm=llm,
        skill_registry=_Registry([_Skill("s1", tools=["tool_a"])]),
        ai_message_type=AIMessage,
        human_message_type=HumanMessage,
        system_message_type=SystemMessage,
        planner_trajectory_type=lambda **kw: kw,
        resolve_effective_skills=lambda _s, _r: ["s1"],
        is_pipeline_internal_ai=lambda _m: False,
        is_ai_with_tool_calls=lambda _m: False,
        strip_think=lambda t: t,
        normalize_toolcalls_list=lambda calls: calls,
        extract_tool_calls_from_jsonish_text=lambda _t: [],
        coerce_content_str=lambda x: x if isinstance(x, str) else str(x),
        canonical_tool_name=lambda n: str(n),
    )

    ai_msg = out["messages"][0]
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "tool_a"
    assert isinstance(out.get("planner_calls_by_subquery"), list)
    assert len(out["planner_calls_by_subquery"]) == 2


def test_execute_planner_node_blocks_tools_outside_skill_scope():
    llm = _PlannerLLM([AIMessage(content="ok", tool_calls=[{"id": "b1", "name": "tool_b", "args": {}}])])
    state = {
        "messages": [HumanMessage(content="haz plan")],
        "analyzer": {"subqueries": ["q1"]},
    }
    out = execute_planner_node(
        state,
        tools=[_Tool("tool_a"), _Tool("tool_b")],
        cfg=type("Cfg", (), {"enable_thinking": True, "max_retries": 0})(),
        planner_llm=llm,
        skill_registry=_Registry([_Skill("s1", tools=["tool_a"])]),
        ai_message_type=AIMessage,
        human_message_type=HumanMessage,
        system_message_type=SystemMessage,
        planner_trajectory_type=lambda **kw: kw,
        resolve_effective_skills=lambda _s, _r: ["s1"],
        is_pipeline_internal_ai=lambda _m: False,
        is_ai_with_tool_calls=lambda _m: False,
        strip_think=lambda t: t,
        normalize_toolcalls_list=lambda calls: calls,
        extract_tool_calls_from_jsonish_text=lambda _t: [],
        coerce_content_str=lambda x: x if isinstance(x, str) else str(x),
        canonical_tool_name=lambda n: str(n),
    )

    ai_msg = out["messages"][0]
    assert ai_msg.tool_calls == []
    assert "No native tool calls" in out["planner_trajs"][0]["description"]
    assert out["planner_calls_by_subquery"][0]["planned_calls"] == 0
    assert out["planner_calls_by_subquery"][0]["skipped_reason"] == "no_tool_calls_generated"


def test_execute_planner_node_builds_subqueries_from_prompt_when_analyzer_missing():
    llm = _PlannerLLM(
        [
            AIMessage(content="ok", tool_calls=[{"id": "c1", "name": "tool_a", "args": {"i": 1}}]),
            AIMessage(content="ok", tool_calls=[{"id": "c2", "name": "tool_a", "args": {"i": 2}}]),
        ]
    )
    state = {
        "messages": [
            HumanMessage(
                content='Concilia: {"credito_id":"LOC-1","saldo_total":10}, {"credito_id":"LOC-2","saldo_total":20}'
            )
        ],
        "analyzer": {},
    }
    out = execute_planner_node(
        state,
        tools=[_Tool("tool_a")],
        cfg=type("Cfg", (), {"enable_thinking": True, "max_retries": 0})(),
        planner_llm=llm,
        skill_registry=_Registry([_Skill("s1", tools=["tool_a"])]),
        ai_message_type=AIMessage,
        human_message_type=HumanMessage,
        system_message_type=SystemMessage,
        planner_trajectory_type=lambda **kw: kw,
        resolve_effective_skills=lambda _s, _r: ["s1"],
        is_pipeline_internal_ai=lambda _m: False,
        is_ai_with_tool_calls=lambda _m: False,
        strip_think=lambda t: t,
        normalize_toolcalls_list=lambda calls: calls,
        extract_tool_calls_from_jsonish_text=lambda _t: [],
        coerce_content_str=lambda x: x if isinstance(x, str) else str(x),
        canonical_tool_name=lambda n: str(n),
    )

    assert len(out["planner_trajs"]) == 2
    assert all(isinstance(tr, dict) for tr in out["planner_trajs"])
    ai_msg = out["messages"][0]
    assert len(ai_msg.tool_calls) == 2


def test_execute_planner_node_aligns_entity_id_to_current_subquery_and_avoids_history_leak():
    class _InspectPlannerLLM:
        def __init__(self):
            self.invocations = []

        def bind_tools(self, _tools):
            return self

        def invoke(self, messages):
            self.invocations.append(messages)
            # Intentionally wrong entity id; planner should align to LOC-0010.
            return AIMessage(
                content="ok",
                tool_calls=[
                    {
                        "id": "z1",
                        "name": "reconcile_credit_accounting",
                        "args": {"credito_id": "LOC-0005", "balance": "39336.16"},
                    }
                ],
            )

    llm = _InspectPlannerLLM()
    previous_turn = HumanMessage(
        content='Realiza la conciliación del Crédito LOC-0005. Estatus: Vigente / Al corriente.'
    )
    state = {
        "messages": [previous_turn, HumanMessage(content="irrelevante para planner")],
        "analyzer": {
            "subqueries": [
                'Realiza la conciliación del crédito {"credito_id":"LOC-0010","saldo_total":39336.16}'
            ]
        },
    }
    out = execute_planner_node(
        state,
        tools=[_Tool("reconcile_credit_accounting")],
        cfg=type("Cfg", (), {"enable_thinking": True, "max_retries": 0})(),
        planner_llm=llm,
        skill_registry=_Registry([_Skill("s1", tools=["reconcile_credit_accounting"])]),
        ai_message_type=AIMessage,
        human_message_type=HumanMessage,
        system_message_type=SystemMessage,
        planner_trajectory_type=lambda **kw: kw,
        resolve_effective_skills=lambda _s, _r: ["s1"],
        is_pipeline_internal_ai=lambda _m: False,
        is_ai_with_tool_calls=lambda _m: False,
        strip_think=lambda t: t,
        normalize_toolcalls_list=lambda calls: calls,
        extract_tool_calls_from_jsonish_text=lambda _t: [],
        coerce_content_str=lambda x: x if isinstance(x, str) else str(x),
        canonical_tool_name=lambda n: str(n),
    )

    ai_msg = out["messages"][0]
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["args"]["credito_id"] == "LOC-0010"
    # Ensure planner invocation did not include prior human turns.
    sent = llm.invocations[0]
    assert len(sent) == 2  # system + current subquery prompt


def test_execute_planner_node_injects_db_path_hint_for_sql_tools():
    llm = _PlannerLLM(
        [
            AIMessage(
                content="ok",
                tool_calls=[{"id": "d1", "name": "nl2sql_sqlite", "args": {"user_request": "consulta"}}],
            )
        ]
    )
    state = {
        "messages": [HumanMessage(content="haz plan")],
        "analyzer": {"subqueries": ["dame informacion del credito LOC-0004 solo transacciones.db"]},
    }
    out = execute_planner_node(
        state,
        tools=[_Tool("nl2sql_sqlite")],
        cfg=type("Cfg", (), {"enable_thinking": True, "max_retries": 0})(),
        planner_llm=llm,
        skill_registry=_Registry([_Skill("s1", tools=["nl2sql_sqlite"])]),
        ai_message_type=AIMessage,
        human_message_type=HumanMessage,
        system_message_type=SystemMessage,
        planner_trajectory_type=lambda **kw: kw,
        resolve_effective_skills=lambda _s, _r: ["s1"],
        is_pipeline_internal_ai=lambda _m: False,
        is_ai_with_tool_calls=lambda _m: False,
        strip_think=lambda t: t,
        normalize_toolcalls_list=lambda calls: calls,
        extract_tool_calls_from_jsonish_text=lambda _t: [],
        coerce_content_str=lambda x: x if isinstance(x, str) else str(x),
        canonical_tool_name=lambda n: str(n),
    )

    ai_msg = out["messages"][0]
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["args"]["db_path"] == "transacciones.db"
    assert ai_msg.tool_calls[0]["args"]["execute"] is True


def test_execute_planner_node_contabilidad_is_deterministic_1_to_1():
    llm = _PlannerLLM([])  # Should not be used in deterministic branch.
    state = {
        "messages": [HumanMessage(content="haz plan")],
        "analyzer": {
            "subqueries": [
                "Realiza la conciliacion del Credito LOC-0005. Estatus: Vigente / Al corriente. Saldo: 19789.9."
            ]
        },
    }
    out = execute_planner_node(
        state,
        tools=[_Tool("query_transactions_db"), _Tool("query_accounting_db")],
        cfg=type("Cfg", (), {"enable_thinking": True, "max_retries": 0})(),
        planner_llm=llm,
        skill_registry=_Registry([_Skill("contabilidad_instantanea", tools=["query_transactions_db", "query_accounting_db"])]),
        ai_message_type=AIMessage,
        human_message_type=HumanMessage,
        system_message_type=SystemMessage,
        planner_trajectory_type=lambda **kw: kw,
        resolve_effective_skills=lambda _s, _r: ["contabilidad_instantanea"],
        is_pipeline_internal_ai=lambda _m: False,
        is_ai_with_tool_calls=lambda _m: False,
        strip_think=lambda t: t,
        normalize_toolcalls_list=lambda calls: calls,
        extract_tool_calls_from_jsonish_text=lambda _t: [],
        coerce_content_str=lambda x: x if isinstance(x, str) else str(x),
        canonical_tool_name=lambda n: str(n),
    )

    ai_msg = out["messages"][0]
    assert len(ai_msg.tool_calls) == 2
    assert ai_msg.tool_calls[0]["name"] == "query_transactions_db"
    assert ai_msg.tool_calls[0]["args"]["query"] == "SELECT tipo, monto FROM movimientos WHERE credito_id = 'LOC-0005'"
    assert ai_msg.tool_calls[1]["name"] == "query_accounting_db"
    assert ai_msg.tool_calls[1]["args"]["query"] == (
        "SELECT saldo_total, estatus, saneamiento_calculado FROM estados_cuenta WHERE credito_id = 'LOC-0005'"
    )
    assert out["planner_calls_by_subquery"][0]["planned_calls"] == 2


def test_execute_planner_node_chat_db_is_deterministic_by_intent():
    llm = _PlannerLLM([])  # Should not be used in deterministic branch.
    state = {
        "messages": [HumanMessage(content="haz plan")],
        "analyzer": {
            "subqueries": ["muestrame el schema de transacciones.db"],
            "subquery_intents": [["explain_schema"]],
        },
    }
    out = execute_planner_node(
        state,
        tools=[_Tool("inspect_sqlite_schema"), _Tool("nl2sql")],
        cfg=type("Cfg", (), {"enable_thinking": True, "max_retries": 0})(),
        planner_llm=llm,
        skill_registry=_Registry(
            [
                _Skill(
                    "chat_db",
                    tools=["inspect_sqlite_schema", "nl2sql"],
                    world="chat_db",
                    intents=["query_data", "explain_schema"],
                    planner_policy={"allowed_dag_patterns": ["deterministic_chat_db_query"]},
                )
            ]
        ),
        ai_message_type=AIMessage,
        human_message_type=HumanMessage,
        system_message_type=SystemMessage,
        planner_trajectory_type=lambda **kw: kw,
        resolve_effective_skills=lambda _s, _r: ["chat_db"],
        is_pipeline_internal_ai=lambda _m: False,
        is_ai_with_tool_calls=lambda _m: False,
        strip_think=lambda t: t,
        normalize_toolcalls_list=lambda calls: calls,
        extract_tool_calls_from_jsonish_text=lambda _t: [],
        coerce_content_str=lambda x: x if isinstance(x, str) else str(x),
        canonical_tool_name=lambda n: str(n),
    )

    ai_msg = out["messages"][0]
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "inspect_sqlite_schema"
    assert ai_msg.tool_calls[0]["args"]["db_path"] == "transacciones.db"
    assert out["planner_trajs"][0]["intent"] == "explain_schema"
