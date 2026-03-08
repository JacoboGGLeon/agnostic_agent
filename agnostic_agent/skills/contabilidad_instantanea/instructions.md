# Contabilidad Instantanea Skill

Use this skill for deterministic, one-credit accounting reconciliation in Spanish.

Primary alignment target:
- Notebook: `examples/finance/notebook_finance_es.ipynb`
- Active skill name in that notebook: `reconcile_accounts`
- Allowed tools: `query_transactions_db`, `query_accounting_db`

Execution contract:
1. Input must include `credito_id`, `estatus`, and `saldo_total`.
2. Build one deterministic query to `transacciones.db` through `query_transactions_db`.
3. Build one deterministic query to `contabilidad.db` through `query_accounting_db`.
4. Compare expected values vs reported accounting values and return:
   - `CUADRADO (100% Match)` when saldo and saneamiento match
   - `DRIFT DETECTADO` otherwise
5. Emit explicit evidence per tool call.

Output style:
- Spanish technical language.
- Brief executive summary first.
- Include differences for saldo and saneamiento with numeric values.
