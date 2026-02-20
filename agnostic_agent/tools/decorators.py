"""
Decoradores para herramientas del Agnostic Agent.

Proporciona un wrapper sobre @tool de LangChain que añade metadata
de registro (mode) para el Rich Context.
"""

from typing import Callable, Optional, Literal
from langchain_core.tools import tool as langchain_tool


def tool(
    func: Optional[Callable] = None,
    *,
    mode: Literal["public", "private"] = "public",
):
    """
    Decorador extendido para herramientas del Agnostic Agent.
    
    Wrapper sobre @tool de LangChain que añade metadata para el Planner.
    
    Args:
        mode: "public" (visible al Planner) o "private" (solo interno)
    
    Usage:
        @tool(mode="public")
        def my_tool(text: str) -> str:
            '''Convierte texto a mayúsculas.'''
            return text.upper()
    """
    def decorator(f: Callable) -> Callable:
        # Primero aplicamos el decorador @tool de LangChain
        langchain_decorated = langchain_tool(f)
        
        # Luego añadimos nuestra metadata
        langchain_decorated._agnostic_metadata = {
            "mode": mode,
        }
        
        return langchain_decorated
    
    # Soporte para uso con y sin paréntesis
    if func is not None:
        return decorator(func)
    return decorator
