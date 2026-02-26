from agnostic_agent.core.pipeline_v2.planner import (
    PlannedCall,
    build_subqueries_from_prompt,
    dedupe_planned_calls,
    flatten_tool_calls_by_subquery,
)


def test_build_subqueries_from_prompt_splits_json_objects():
    prompt = (
        'Concilia estos creditos: {"credito_id":"LOC-1","saldo":10}, '
        '{"credito_id":"LOC-2","saldo":20}'
    )
    subqueries = build_subqueries_from_prompt(prompt)
    assert len(subqueries) == 2
    assert "LOC-1" in subqueries[0]
    assert "LOC-2" in subqueries[1]


def test_dedupe_planned_calls_removes_duplicates():
    calls = [
        PlannedCall(name="tool_a", args={"x": 1}, subquery_idx=1),
        PlannedCall(name="tool_a", args={"x": 1}, subquery_idx=1),
        PlannedCall(name="tool_a", args={"x": 1}, subquery_idx=2),
    ]
    deduped = dedupe_planned_calls(calls)
    assert len(deduped) == 2


def test_flatten_tool_calls_by_subquery_normalizes_rows():
    rows = flatten_tool_calls_by_subquery(
        [
            {
                "subquery_idx": 1,
                "tool_calls": [{"name": "tool_a", "args": {"x": 1}}],
            },
            {
                "subquery_idx": 2,
                "tool_calls": [{"name": "tool_b", "args": {"y": 2}}],
            },
        ]
    )
    assert len(rows) == 2
    assert rows[0].name == "tool_a"
    assert rows[1].subquery_idx == 2
