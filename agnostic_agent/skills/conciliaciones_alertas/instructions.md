# Conciliaciones Alertas

Usa esta skill cuando la peticion trate sobre conciliaciones, alertas de variacion, control inventario vs contable, remediacion o incidencias recurrentes.

Intents principales:
- `logical_reconciliation`: concilia un credito puntual con `reconcile_credit_accounting`.
- `inventory_vs_accounting`: compara inventario y vista contable con `nl2sql`.
- `detect_significant_variations`: busca variaciones significativas o atipicas con `nl2sql`.
- `propose_remediation`: propone remediacion o escalamiento usando `reconcile_credit_accounting` y `finance_sources_status`.
- `analyze_recurring_incidents`: analiza incidencias recurrentes usando `nl2sql` y `finance_sources_status`.

Politica operativa:
- Preferir `reconcile_credit_accounting` cuando la peticion sea puntual por `credito_id`.
- Usar `nl2sql` para hallazgos agregados, control de calidad y alertas historicas.
- Usar `finance_sources_status` para validar estado de fuentes antes de escalar o remediar.
