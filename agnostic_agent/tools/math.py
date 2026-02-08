from typing import List, Any
import ast
import operator as _op
import numbers
from agnostic_agent.tools.decorators import tool

# ─────────────────────────────────────────────
# TOOLS matemáticas (evaluadas en Python)
# ─────────────────────────────────────────────

# Operadores permitidos en la expresión matemática
_ALLOWED_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.Mod: _op.mod,
}


def _eval_ast(node: ast.AST) -> float:
    """Evalúa de forma segura un AST restringido a operaciones aritméticas básicas."""
    if isinstance(node, ast.Num):  # Python <3.8
        return node.n
    if isinstance(node, ast.Constant):  # números en 3.8+
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Sólo se permiten números en las constantes.")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Operador no permitido: {op_type.__name__}")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _ALLOWED_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand)

    raise ValueError(f"Nodo de AST no permitido: {type(node).__name__}")


@tool(mode="public", output_schema={"type": "number"})
def eval_math_expression(expression: str) -> float:
    """
    Evalúa una expresión matemática sencilla usando Python de forma segura.

    Soporta:
      - suma, resta, multiplicación, división, módulo, potencias
      - paréntesis
      - signos unarios (p.ej. -3)

    Ejemplos válidos:
      "1 + 2 * 3"
      "(10 - 4) / 2"
      "2**3 + 5"

    NOTA:
      El operador de potencia soportado es **, NO ^ (que en Python es XOR).
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_ast(parsed.body)
        return float(result)
    except Exception as exc:
        raise ValueError(
            f"No se pudo evaluar la expresión: {expression!r}. Error: {exc}"
        ) from exc


# ─────────────────────────────────────────────
# Helpers numéricos robustos para sum/avg
# ─────────────────────────────────────────────

def _coerce_to_float(x: Any) -> float:
    """
    Intenta convertir un valor genérico a float.

    Soporta:
    - ints/floats/np.number
    - strings numéricos ("3.14")
    - dicts con claves típicas: "value", "val", "number", "num"
    """
    if isinstance(x, numbers.Number):
        return float(x)

    if isinstance(x, str):
        # Permite strings como "3.14", "42"
        return float(x.strip())

    if isinstance(x, dict):
        for key in ("value", "val", "number", "num"):
            if key in x:
                return _coerce_to_float(x[key])

    raise ValueError(f"No se pudo interpretar {x!r} como número.")


@tool(mode="public", output_schema={"type": "number"})
def sum_numbers(numbers: List[Any]) -> float:
    """
    Devuelve la suma de una lista de números.

    La tool es robusta: acepta tanto números puros como estructuras
    que contengan el número, por ejemplo:

      - [1, 2.5, "3"]
      - [{"value": 10}, {"number": "20"}]

    Para evitar errores de validación Pydantic, el tipo es List[Any]
    y se hace coerción interna a float.
    """
    if not isinstance(numbers, list):
        raise ValueError("El parámetro 'numbers' debe ser una lista.")

    vals = [_coerce_to_float(n) for n in numbers]
    return float(sum(vals))


@tool(mode="public", output_schema={"type": "number"})
def average_numbers(numbers: List[Any]) -> float:
    """
    Devuelve la media aritmética de una lista de números.

    Mismo comportamiento robusto que sum_numbers:
      - [1, 2.5, "3"]
      - [{"value": 10}, {"number": "20"}]
    """
    if not isinstance(numbers, list):
        raise ValueError("El parámetro 'numbers' debe ser una lista.")
    if not numbers:
        raise ValueError("La lista de números está vacía.")

    vals = [_coerce_to_float(n) for n in numbers]
    return float(sum(vals) / len(vals))
