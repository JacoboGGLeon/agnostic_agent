# Semantic Researcher

Usa este mundo para retrieval y sintesis basada en evidencia documental.

Intents principales:
- `semantic_lookup`: recuperar evidencia relevante desde las fuentes declaradas.
- `semantic_synthesis`: sintetizar evidencia recuperada con foco en claridad y citas.

Pipeline base:
- `list_knowledge_sources` para inspeccionar fuentes disponibles.
- `search_knowledge_base(top_k=15)` para recuperar evidencia inicial.
- `rerank_docs(top_n=3)` para priorizar los mejores fragmentos.
- Sintesis final con citas y foco en evidencia.

Politica:
- Mantener respuestas apoyadas en evidencia recuperada.
- Priorizar `search_knowledge_base` y `rerank_docs` antes de sintetizar.
- Usar `list_knowledge_sources` cuando la peticion trate sobre cobertura o fuentes disponibles.
