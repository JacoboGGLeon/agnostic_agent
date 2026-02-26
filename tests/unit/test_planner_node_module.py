from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agnostic_agent.graph.planner_node import execute_planner_node


class _Skill:
    def __init__(self, name, tools=None, knowledge=None):
        self.name = name
        self.tools = tools or []
        self.knowledge = knowledge or []
        self.description = name


class _Registry:
    def __init__(self, skills):
        self._skills = {s.name: s for s in skills}

    def get_skill(self, name):
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
