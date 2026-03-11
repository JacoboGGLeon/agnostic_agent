# Costo Saneamiento Contrato

Usa esta skill cuando la peticion trate sobre costo por saneamiento a nivel contrato, liberaciones, dotaciones, resultados mensuales o trazabilidad del costo por credito.

Intents principales:
- `identify_liberaciones_dotaciones`: identifica flujos de liberacion y dotacion sobre un `credito_id` usando `reconcile_credit_accounting`.
- `calculate_monthly_saneamiento_cost`: estima resultados mensuales o cortes agregados con `nl2sql`.
- `contract_traceability`: explica la trazabilidad del costo por contrato con `reconcile_credit_accounting` y `get_saneamiento_rate`.
- `explain_saneamiento_cost`: explica como se obtuvo el costo de saneamiento por credito con `reconcile_credit_accounting`, `get_saneamiento_rate` y `lookup_finance_rule`.

Politica operativa:
- Preferir `reconcile_credit_accounting` para cualquier analisis puntual por `credito_id`.
- Usar `get_saneamiento_rate` para justificar la tasa aplicada al saneamiento.
- Usar `lookup_finance_rule` para explicar la regla de negocio detras del costo.
- Usar `nl2sql` solo para analisis agregados o resultados mensuales read-only.
