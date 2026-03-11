from agnostic_agent.graph.summarization import build_agnostic_user_answer


def test_build_agnostic_user_answer_finance_reconciliation_natural():
    runs = [
        {
            "name": "query_transactions_db",
            "args": {"query": "SELECT tipo, monto FROM movimientos WHERE credito_id = 'LOC-0005'"},
            "output": '{"normalized_query": "SELECT tipo, monto FROM movimientos WHERE credito_id = \'LOC-0005\'", "columns": ["tipo", "monto"], "rows": [["DESEMBOLSO", 23986.48], ["PAGO", 4196.58]]}',
        },
        {
            "name": "query_accounting_db",
            "args": {"query": "SELECT saldo_total, estatus, saneamiento_calculado FROM estados_cuenta WHERE credito_id = 'LOC-0005'"},
            "output": '{"normalized_query": "SELECT saldo_total, estatus, saneamiento_calculado FROM estados_cuenta WHERE credito_id = \'LOC-0005\'", "columns": ["saldo_total", "estatus", "saneamiento_calculado"], "rows": [[19789.9, "Vigente / Al corriente", 197.9]]}',
        },
    ]

    text = build_agnostic_user_answer(
        "Realiza la conciliacion del Credito LOC-0005. Estatus: Vigente / Al corriente. Saldo: 19789.9.",
        runs,
    )

    assert "Conciliacion del credito LOC-0005" in text
    assert "CUADRADO (100% Match)" in text
    assert "Saldo esperado" in text
    assert "Reserva esperada" in text


def test_build_agnostic_user_answer_finance_batch_all_requested_credits():
    runs = [
        {
            "name": "query_transactions_db",
            "args": {"query": "SELECT tipo, monto FROM movimientos WHERE credito_id = 'LOC-0004'"},
            "output": '{"columns": ["tipo", "monto"], "rows": [["DESEMBOLSO", 100.0], ["PAGO", 20.0]]}',
        },
        {
            "name": "query_accounting_db",
            "args": {"query": "SELECT saldo_total, estatus, saneamiento_calculado FROM estados_cuenta WHERE credito_id = 'LOC-0004'"},
            "output": '{"columns": ["saldo_total", "estatus", "saneamiento_calculado"], "rows": [[80.0, "Desembolsado", 0.8]]}',
        },
        {
            "name": "query_transactions_db",
            "args": {"query": "SELECT tipo, monto FROM movimientos WHERE credito_id = 'LOC-0005'"},
            "output": '{"columns": ["tipo", "monto"], "rows": [["DESEMBOLSO", 200.0], ["PAGO", 50.0]]}',
        },
        {
            "name": "query_accounting_db",
            "args": {"query": "SELECT saldo_total, estatus, saneamiento_calculado FROM estados_cuenta WHERE credito_id = 'LOC-0005'"},
            "output": '{"columns": ["saldo_total", "estatus", "saneamiento_calculado"], "rows": [[150.0, "Vigente / Al corriente", 1.5]]}',
        },
    ]
    prompt = (
        'Realiza la conciliación de los siguientes créditos: '
        '{"credito_id":"LOC-0004","estatus":"Desembolsado","saldo_total":80.0}, '
        '{"credito_id":"LOC-0005","estatus":"Vigente / Al corriente","saldo_total":150.0}'
    )
    text = build_agnostic_user_answer(
        prompt,
        runs,
        [
            'Realiza la conciliación: {"credito_id":"LOC-0004"}',
            'Realiza la conciliación: {"credito_id":"LOC-0005"}',
        ],
    )
    assert "Conciliacion del credito LOC-0004" in text
    assert "Conciliacion del credito LOC-0005" in text
    assert text.count("CUADRADO (100% Match)") >= 2


def test_build_agnostic_user_answer_detailed_reconciliation_from_direct_tool():
    runs = [
        {
            "name": "reconcile_credit_accounting",
            "args": {"credito_id": "LOC-0004"},
            "output": {
                "ok": True,
                "credito_id": "LOC-0004",
                "estatus": "Desembolsado",
                "status": "CUADRADO (100% Match)",
                "flujos": {"DESEMBOLSO": 35451.48, "PAGO": 6010.84, "PENALIZACION": 0.0, "DESCUENTO": 0.0},
                "saldo": {"reportado": 29440.64, "esperado": 29440.64, "diferencia": -0.0},
                "saneamiento": {"tasa": 0.01, "reportado": 294.41, "esperado": 294.41, "diferencia": 0.0},
            },
        }
    ]

    text = build_agnostic_user_answer(
        "explicame detalladamente como llegaste a esto: LOC-0004",
        runs,
        ['explicame detalladamente como llegaste a esto: {"credito_id":"LOC-0004"}'],
    )

    assert "Conciliacion del credito LOC-0004" in text
    assert "Los flujos considerados fueron" in text
    assert "saldo esperado" in text.lower()
    assert "saneamiento esperado" in text.lower()


def test_build_agnostic_user_answer_finance_batch_direct_tool_is_detailed():
    runs = [
        {
            "name": "reconcile_credit_accounting",
            "args": {"credito_id": "LOC-0004"},
            "output": {
                "ok": True,
                "credito_id": "LOC-0004",
                "estatus": "Desembolsado",
                "status": "CUADRADO (100% Match)",
                "flujos": {"DESEMBOLSO": 100.0, "PAGO": 20.0, "PENALIZACION": 0.0, "DESCUENTO": 0.0},
                "saldo": {"reportado": 80.0, "esperado": 80.0, "diferencia": 0.0},
                "saneamiento": {"tasa": 0.01, "reportado": 0.8, "esperado": 0.8, "diferencia": 0.0},
            },
        },
        {
            "name": "reconcile_credit_accounting",
            "args": {"credito_id": "LOC-0005"},
            "output": {
                "ok": True,
                "credito_id": "LOC-0005",
                "estatus": "Vigente / Al corriente",
                "status": "CUADRADO (100% Match)",
                "flujos": {"DESEMBOLSO": 200.0, "PAGO": 50.0, "PENALIZACION": 0.0, "DESCUENTO": 0.0},
                "saldo": {"reportado": 150.0, "esperado": 150.0, "diferencia": 0.0},
                "saneamiento": {"tasa": 0.01, "reportado": 1.5, "esperado": 1.5, "diferencia": 0.0},
            },
        },
    ]

    text = build_agnostic_user_answer(
        'Realiza la conciliación de los siguientes créditos: {"credito_id":"LOC-0004"}, {"credito_id":"LOC-0005"}',
        runs,
        ['Realiza la conciliación: {"credito_id":"LOC-0004"}', 'Realiza la conciliación: {"credito_id":"LOC-0005"}'],
    )

    assert "Realice la conciliacion de 2 credito(s)." in text
    assert "Conciliacion del credito LOC-0004" in text
    assert "Conciliacion del credito LOC-0005" in text
    assert "Todos quedaron CUADRADOS." in text
