from agnostic_agent.graph.summarization import build_agnostic_user_answer


def test_build_agnostic_user_answer_mentions_missing_credit_filter_for_nl2sql():
    runs = [
        {
            "name": "nl2sql_sqlite",
            "args": {
                "user_request": "dame informacion sobre el credito LOC-0004",
                "entity_id": "LOC-0004",
            },
            "output": {
                "ok": True,
                "db_path": "session/embeddings.db",
                "user_request": "dame informacion sobre el credito LOC-0004",
                "entity_id": "LOC-0004",
                "chosen_table": "chunks_meta",
                "where_clauses": [],
                "execution": {"ok": True, "row_count": 50, "rows": [[1, "a"]]},
            },
        }
    ]

    text = build_agnostic_user_answer("dame informacion sobre el credito LOC-0004", runs)

    assert "No pude filtrar por LOC-0004" in text
    assert "contabilidad.db" in text
