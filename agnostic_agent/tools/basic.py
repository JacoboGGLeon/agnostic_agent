from __future__ import annotations

from agnostic_agent.tools.decorators import tool


@tool(mode="public", output_schema={"type": "string"})
def to_upper(text: str) -> str:
    """Convierte el texto a mayúsculas."""
    return text.upper()


@tool(mode="public", output_schema={"type": "integer"})
def word_count(text: str) -> int:
    """Cuenta el número de palabras en el texto."""
    return len(text.split())


@tool(mode="public", output_schema={"type": "boolean"})
def is_palindrome(text: str) -> bool:
    """Verifica si el texto es un palíndromo."""
    clean = "".join(c.lower() for c in text if c.isalnum())
    return clean == clean[::-1]
