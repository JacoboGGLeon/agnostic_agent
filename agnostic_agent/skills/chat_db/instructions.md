# Chat DB

Usa este mundo cuando el usuario quiera consultar, explorar o explicar datos estructurados en SQLite.

Politica:
- Preferir `inspect_sqlite_schema` para preguntas de estructura.
- Preferir `nl2sql` para preguntas de datos, agregacion o comparacion.
- Mantener siempre modo read-only.
- Si la consulta viene en lote, descomponer por subquery y resolver cada una dentro del mismo mundo.
