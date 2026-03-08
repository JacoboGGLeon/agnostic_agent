name: reconcile_accounts
description: Realiza la conciliacion contable de un credito ("Cuadre") verificando flujos vs contabilidad y calculo de reservas.
version: 1.0.0
tools:
  - query_transactions_db
  - query_accounting_db
inputs:
  - name: credito_id
    description: ID del credito a conciliar (ej. LOC-0001)
    required: true
steps: |
  PASO 1: Obtener Flujos (Universo 1)
  Ejecuta `query_transactions_db` para obtener todos los movimientos:
  "SELECT tipo, monto FROM movimientos WHERE credito_id = '{credito_id}'"
  Luego suma los montos por tipo (Total Desembolsos, Total Pagos, Total Penalizaciones, Total Descuentos).

  PASO 2: Obtener Contabilidad (Universo 2)
  Ejecuta `query_accounting_db` para obtener el estado actual:
  "SELECT saldo_total, estatus, saneamiento_calculado FROM estados_cuenta WHERE credito_id = '{credito_id}'"

  PASO 3: Verificar Ecuacion de Saldos
  Calcula el Saldo Esperado: (Desembolsos - Pagos) + Penalizaciones - Descuentos.
  Compara Saldo Esperado vs Saldo Total (reportado).
  *DEBEN SER IDENTICOS* (Diferencia < 0.01).

  PASO 4: Verificar Saneamiento
  Consulta la KB (Reglas de Negocio) para obtener la Tasa de Saneamiento segun el 'estatus'.
  Calcula: Reserva Esperada = Saldo Total * Tasa.
  Compara Reserva Esperada vs 'saneamiento_calculado' (reportado).

  PASO 5: Reportar Resultado
  - Si ambas verificaciones son correctas, reporta: "CUADRADO (100% Match)".
  - Si hay diferencias, reporta "DRIFT DETECTADO" y explica donde (diferencia en saldo o en reserva).
