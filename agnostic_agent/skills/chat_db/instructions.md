# Chat DB

Usa este mundo cuando el usuario quiera consultar, explorar o explicar datos estructurados en SQLite.

Capacidades:
- `query_data`: consultas de registros y filtros.
- `aggregate_data`: agregaciones y agrupaciones.
- `compare_entities`: comparaciones y rankings.
- `explain_schema`: inspeccion de estructura con `inspect_sqlite_schema`.
- `batch_query`: descomposicion por subquery cuando la peticion llega en lote.

Politica:
- Preferir `inspect_sqlite_schema` para preguntas de estructura.
- Preferir `nl2sql` para preguntas de datos, agregacion o comparacion.
- Mantener siempre modo read-only.
- `execute_sql_readonly` es una primitive de apoyo; no debe usarse para escritura.
