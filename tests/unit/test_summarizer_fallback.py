from agnostic_agent.logic import summarize_tool_runs_compact


def test_summarize_tool_runs_compact_is_non_empty_and_concise():
    runs = [
        {
            "name": "reconcile_credit_accounting",
            "args": {"credito_id": "LOC-0004", "balance": "29440.64"},
            "output": {"ok": True, "credito_id": "LOC-0004", "status": "CUADRADO (100% Match)"},
        },
        {
            "name": "get_saneamiento_rate",
            "args": {"estatus": "Mora media (31-60 dias)"},
            "output": {"found": True, "tasa_saneamiento": 0.2},
        },
    ]

    text = summarize_tool_runs_compact(runs)

    assert text.strip()
    assert "Se ejecutaron 2 tools." in text
    assert "reconcile_credit_accounting" in text
    assert "LOC-0004" in text
