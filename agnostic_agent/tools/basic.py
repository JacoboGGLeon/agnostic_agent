from langchain_core.tools import tool

@tool
def to_upper(text: str) -> str:
    """Convierte el texto a MAYÚSCULAS."""
    return text.upper()


@tool
def word_count(text: str) -> int:
    """Devuelve el número de palabras en el texto."""
    return len([w for w in text.split() if w])


@tool
def is_palindrome(text: str) -> bool:
    """True si el texto (sin espacios/casos) es palíndromo."""
    s = "".join(ch.lower() for ch in text if ch.isalnum())
    return s == s[::-1]
