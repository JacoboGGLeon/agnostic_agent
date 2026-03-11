# Contabilidad Automatica

Usa este mundo cuando la peticion trate sobre conciliacion, auditoria o exploracion de datos contables.

Politica:
- Preferir `reconcile_credit_accounting` para conciliacion puntual por credito.
- Usar `nl2sql` para exploracion financiera ad-hoc.
- Consultar `get_saneamiento_rate` para explicar reglas o auditar reservas.
- Si la peticion viene en lote, generar una subquery por credito.
