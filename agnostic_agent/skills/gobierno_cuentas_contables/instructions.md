# Gobierno Cuentas Contables

Usa esta skill cuando la peticion trate sobre asignacion de cuenta contable, validacion normativa, propuestas de asiento o entrega batch para contabilizacion.

Intents principales:
- `assign_accounting_account`: asigna o propone cuenta contable segun atributos del contrato usando `nl2sql` y `lookup_finance_dictionary`.
- `validate_accounting_compliance`: valida cumplimiento contable usando `nl2sql`, `lookup_finance_rule` y `finance_sources_status`.
- `generate_accounting_entry`: genera una propuesta de asiento con `nl2sql`, `get_saneamiento_rate` y `lookup_finance_rule`.
- `export_accounting_batch`: prepara payloads batch para aplicativos con `nl2sql` y `finance_sources_status`.

Politica operativa:
- Usar `nl2sql` para leer contratos, estatus y atributos contables en modo read-only.
- Usar `lookup_finance_dictionary` para explicar campos y cuentas.
- Usar `lookup_finance_rule` y `get_saneamiento_rate` para justificar reglas contables y tasas.
- Usar `finance_sources_status` cuando se requiera validar que las fuentes esten listas para entrega batch.
