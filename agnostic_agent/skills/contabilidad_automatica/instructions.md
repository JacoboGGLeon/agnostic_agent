# Contabilidad Automatica

Usa este mundo cuando la peticion trate sobre conciliacion, auditoria, reglas de saneamiento o exploracion financiera sobre transacciones, contabilidad, reglas y diccionario.

Intents principales:
- `reconcile_credit`: conciliacion puntual por `credito_id` con `reconcile_credit_accounting`.
- `audit_drift`: auditoria de drift y reserva sobre un credito.
- `query_financial_data`: exploracion ad-hoc con `nl2sql` sobre `contabilidad.db` y `transacciones.db`.
- `explain_rule`: explicacion de reglas de saneamiento por `estatus` con `get_saneamiento_rate` y `lookup_finance_rule`.
- `batch_reconcile`: una subquery por credito cuando la peticion viene en lote.
- `explain_reconciliation_result`: explica paso a paso como se obtuvo un resultado de conciliacion para un `credito_id`.
- `explain_reconciliation_flows`: explica los flujos financieros (desembolso, pago, penalizacion, descuento) que sustentan la conciliacion.

Politica operativa:
- Preferir `reconcile_credit_accounting` para conciliacion y auditoria por credito.
- Usar `nl2sql` solo para exploracion financiera read-only.
- Para reglas, resolver `estatus` y usar `get_saneamiento_rate`; complementar con `lookup_finance_rule`.
- Para terminos de negocio, usar `lookup_finance_dictionary`.
- Consultar `finance_sources_status` cuando haga falta verificar fuentes.
- Para explicar una conciliacion ya obtenida, reutilizar `reconcile_credit_accounting` y responder grounded con flujos, saldo esperado/reportado y saneamiento esperado/reportado.
