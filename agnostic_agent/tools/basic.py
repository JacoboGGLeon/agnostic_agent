from __future__ import annotations

from agnostic_agent.tools.decorators import tool


@tool(mode="public")
def to_upper(text: str) -> str:
    """
    Convierte el texto a mayúsculas.
    
    ### Ejemplo Teórico-Práctico
    
    Sea $x$ el texto de entrada y $f(x)$ la función de transformación:
    
    $$ f(x) = x.upper() $$
    
    Donde:
    - $x$: "hola mundo"
    - Resultado: "HOLA MUNDO"
    """
    return text.upper()


@tool(mode="public")
def word_count(text: str) -> int:
    """Cuenta el número de palabras en el texto."""
    return len(text.split())


@tool(mode="public")
def is_palindrome(text: str) -> bool:
    """Verifica si el texto es un palíndromo."""
    clean = "".join(c.lower() for c in text if c.isalnum())
    return clean == clean[::-1]
