# Analisis Saneamiento Stage

Usa esta skill cuando la peticion trate sobre despliegues generales de cartera, explicacion de deterioro o mejora, tendencias, proyecciones o soporte explicativo sobre incidencias.

Intents principales:
- `portfolio_breakdown`: despliega la cartera por dimensiones usando `nl2sql`.
- `explain_deterioration`: explica deterioro o mejora de un `credito_id` usando `reconcile_credit_accounting`, `get_saneamiento_rate` y `lookup_finance_rule`.
- `generate_trends_and_projections`: genera tendencias y proyecciones con `nl2sql`.
- `support_and_fix_incidents`: responde preguntas de soporte usando `nl2sql`, `lookup_finance_dictionary` y `finance_sources_status`.

Politica operativa:
- Usar `nl2sql` para cortes analiticos, reporting y proyecciones read-only.
- Usar `reconcile_credit_accounting` para explicar deterioro o mejora sobre un credito puntual.
- Usar `get_saneamiento_rate` y `lookup_finance_rule` para justificar cambios de stage o deterioro.
- Usar `lookup_finance_dictionary` y `finance_sources_status` para soporte e incidencias.
