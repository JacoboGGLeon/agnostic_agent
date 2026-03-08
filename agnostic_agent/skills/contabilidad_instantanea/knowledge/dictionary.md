# Diccionario de Datos: Contabilidad Instantánea

## Universo 1: Transacciones (Bancarias / Core)
Fuente de la verdad para el flujo de efectivo.
Archivo: `transacciones.db`
Tabla: `movimientos`

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | TEXT | Identificador único de la transacción (TXN-XXXX). |
| `credito_id` | TEXT | Identificador del crédito asociado (LOC-XXXX). |
| `tipo` | TEXT | Tipo de movimiento: `DESEMBOLSO`, `PAGO`, `PENALIZACION`, `DESCUENTO`. |
| `monto` | REAL | Cantidad monetaria de la transacción. |
| `fecha` | TEXT | Fecha de la transacción (YYYY-MM-DD). |
| `descripcion` | TEXT | Detalle narrativo de la operación. |

## Universo 2: Contabilidad (Saldos y Provisiones)
Reporte del estado actual del crédito y su reserva de riesgo.
Archivo: `contabilidad.db`
Tabla: `estados_cuenta`

| Columna | Tipo | Descripción |
|---|---|---|
| `credito_id` | TEXT | Identificador del crédito (Clave Primaria). |
| `cliente_id` | TEXT | Identificador del cliente. |
| `estatus` | TEXT | Estado actual del crédito (Ver Reglas de Negocio). |
| `saldo_total` | REAL | Saldo reportado al corte. |
| `capital_vigente` | REAL | Parte del saldo que es capital (en este modelo es el total). |
| `saneamiento_calculado`| REAL | Reserva de riesgo calculada según el estatus (`saldo_total * tasa`). |
