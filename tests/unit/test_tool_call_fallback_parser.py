from langchain_core.messages import AIMessage

from agnostic_agent.logic import extract_tool_calls


def test_extract_tool_calls_from_tool_uses_json_blocks():
    msg = AIMessage(
        content="""
--- Plan 1 ---
{
  "tool_uses": [
    {
      "recipient_name": "functions.reconcile_credit_accounting",
      "parameters": {"credito_id": "LOC-0004"}
    }
  ]
}
--- Plan 2 ---
{
  "tool_uses": [
    {
      "recipient_name": "functions.query_transactions_db",
      "parameters": {"credito_id": "LOC-0005"}
    }
  ]
}
""".strip()
    )

    calls = extract_tool_calls(msg)

    names = [c["name"] for c in calls]
    assert "reconcile_credit_accounting" in names
    assert "query_transactions_db" in names
    assert len(calls) == 2


def test_extract_tool_calls_deduplicates_same_call():
    msg = AIMessage(
        content="""
{
  "tool_uses": [
    {
      "recipient_name": "functions.reconcile_credit_accounting",
      "parameters": {"credito_id": "LOC-0004"}
    }
  ]
}
{
  "tool_uses": [
    {
      "recipient_name": "functions.reconcile_credit_accounting",
      "parameters": {"credito_id": "LOC-0004"}
    }
  ]
}
""".strip()
    )

    calls = extract_tool_calls(msg)

    assert len(calls) == 1
    assert calls[0]["name"] == "reconcile_credit_accounting"
    assert calls[0]["args"] == {"credito_id": "LOC-0004"}
