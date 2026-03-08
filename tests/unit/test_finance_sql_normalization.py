import os

import pytest


@pytest.mark.unit
def test_query_accounting_db_normalizes_common_alias_columns():
    acc_db = os.path.join(os.getcwd(), "session", "contabilidad.db")
    if not os.path.exists(acc_db):
        pytest.skip("session contabilidad.db not present in this workspace")

    from agnostic_agent.tools.finance import query_accounting_db

    out = query_accounting_db.invoke(
        {
            "query": "SELECT * FROM creditos WHERE codigo = 'LOC-0005' AND estatus = 'Vigente / Al corriente' AND saldo = 19789.9"
        }
    )

    assert isinstance(out, str)
    assert not out.startswith("Error SQL: no such column")
