import os

import pytest


@pytest.mark.unit
def test_nl2sql_prefers_skill_knowledge_dbs_for_finance_queries():
    from agnostic_agent.tools.introspection import _resolve_sqlite_db_candidates

    candidates = _resolve_sqlite_db_candidates("", user_request="dame informacion del credito LOC-0004")
    assert candidates, "Expected at least one sqlite candidate"
    basenames = [os.path.basename(p).lower() for p in candidates]
    assert "contabilidad.db" in basenames
    assert "transacciones.db" in basenames
    assert "embeddings.db" not in basenames
