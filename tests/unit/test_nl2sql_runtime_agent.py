from agnostic_agent.tools.nl2sql_runtime import NL2SQLRuntimeAgent, NL2SQLRuntimeConfig


def test_nl2sql_runtime_agent_runs_end_to_end_on_catalog():
    agent = NL2SQLRuntimeAgent(
        NL2SQLRuntimeConfig(
            catalog_path="agnostic_agent/skills/chat_db/knowledge/catalog_contabilidad.json",
            db_path="agnostic_agent/skills/nl2sql_sqlite/knowledge/contabilidad.db",
            row_limit=20,
            k=4,
        )
    )

    out = agent.query(
        user_query="dame informacion del credito",
        execute=False,
        entity_id="LOC-0004",
    )

    assert out.get("ok") is True
    assert out.get("agent") == "nl2sql_runtime_v4"
    assert str(out.get("catalog_path", "")).lower().endswith("catalog_contabilidad.json")
    assert isinstance(out.get("plan"), dict)
    assert isinstance(out.get("output"), dict)
    assert "generated_sql" in out
