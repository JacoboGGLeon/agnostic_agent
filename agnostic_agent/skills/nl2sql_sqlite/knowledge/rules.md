# Reglas de Negocio: Contabilidad y Saneamiento

## 1. Ecuación de Cuadre (Reconciliación)
Para que un crédito esté "Cuadrado" (100% Match), debe cumplirse estrictamente la siguiente ecuación:

**Saldo Esperado** = `(Total Desembolsos - Total Pagos) + Total Penalizaciones - Total Descuentos`

El Agente debe calcular el `Saldo Esperado` sumando los registros en `transacciones.db` y compararlo con el `saldo_total` reportado en `contabilidad.db`.

## 2. Reglas de Saneamiento (Provisioning)
La reserva ("Saneamiento") se calcula científicamente como un porcentaje del `Saldo Total` dependiendo del `Estatus` del crédito.

| Estatus | Días de Mora | Tasa de Saneamiento (Reserva) |
|---|---|---|
| Solicitud | N/A | 0.00% |
| En análisis | N/A | 0.00% |
| Aprobado | N/A | 0.00% |
| Desembolsado | 0 | 1.00% |
| Vigente / Al corriente | 0 | 1.00% |
| Mora temprana (1–30 días) | 1-30 | 5.00% |
| Mora media (31–60 días) | 31-60 | 20.00% |
| Mora tardía (61–90 días) | 61-90 | 50.00% |
| Cartera vencida (+90 días) | 91+ | 100.00% |
| Castigado / Incobrable | N/A | 100.00% |
| En cobranza externa / Legal | N/A | 100.00% |
| Liquidado / Cerrado | N/A | 0.00% |

**Fórmula de Validación Saneamiento:**
`Reserva Calculada == Saldo Total * Tasa de Saneamiento`
