import os

import pytest


@pytest.mark.unit
def test_nl2sql_agent_sqlite_executes_on_session_dbs():
    # These DBs are part of this repo's typical local workflow (session/).
    acc_db = os.path.join(os.getcwd(), "session", "contabilidad.db")
    tx_db = os.path.join(os.getcwd(), "session", "transacciones.db")
    if not (os.path.exists(acc_db) and os.path.exists(tx_db)):
        pytest.skip("session DBs not present in this workspace")

    from agnostic_agent.tools.introspection import nl2sql_agent_sqlite

    out1 = nl2sql_agent_sqlite.invoke(
        {
            "user_request": "dame informacion del credito LOC-0004",
            "db_path": acc_db,
            "execute": True,
        }
    )
    assert out1.get("ok") is True
    assert out1.get("chosen_table")
    assert "select" in (out1.get("generated_sql") or "").lower()
    assert isinstance(out1.get("execution"), dict) and out1["execution"].get("ok") is True
    assert out1["execution"].get("row_count", 0) >= 1

    out2 = nl2sql_agent_sqlite.invoke(
        {
            "user_request": "muestrame transacciones del credito LOC-0004",
            "db_path": tx_db,
            "execute": True,
        }
    )
    assert out2.get("ok") is True
    assert out2.get("chosen_table")
    assert isinstance(out2.get("execution"), dict) and out2["execution"].get("ok") is True


@pytest.mark.unit
def test_nl2sql_sqlite_uses_entity_filters_and_finance_default_db():
    acc_db = os.path.join(os.getcwd(), "session", "contabilidad.db")
    tx_db = os.path.join(os.getcwd(), "session", "transacciones.db")
    if not (os.path.exists(acc_db) and os.path.exists(tx_db)):
        pytest.skip("session DBs not present in this workspace")

    from agnostic_agent.tools.introspection import nl2sql_sqlite

    out = nl2sql_sqlite.invoke(
        {
            "user_request": "dame informacion del credito",
            "entity_id": "LOC-0004",
            "execute": False,
        }
    )
    assert out.get("ok") is True
    assert str(out.get("db_path", "")).lower().endswith("contabilidad.db")
    assert "credito_id = 'LOC-0004'" in (out.get("generated_sql") or "")
    where = out.get("where_clauses") or []
    assert any("LOC-0004" in str(item) for item in where)

