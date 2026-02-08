"""
Decoradores para herramientas del Agnostic Agent.

Proporciona un wrapper sobre @tool de LangChain que añade metadata 
estructurada (modo, input schema, output schema) para el Rich Context.
"""

from typing import Any, Callable, Optional, Literal
from functools import wraps
from langchain_core.tools import tool as langchain_tool


def tool(
    func: Optional[Callable] = None,
    *,
    mode: Literal["public", "private"] = "public",
    input_schema: Optional[dict] = None,
    output_schema: Optional[dict] = None,
):
    """
    Decorador extendido para herramientas del Agnostic Agent.
    
    Wrapper sobre @tool de LangChain que añade metadata para el Planner.
    
    Args:
        mode: "public" (visible al Planner) o "private" (solo interno)
        input_schema: Esquema JSON de entrada (opcional, se infiere si no se provee)
        output_schema: Esquema JSON de salida (opcional)
    
    Usage:
        @tool(mode="public", output_schema={"type": "string"})
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
            "input_schema": input_schema,
            "output_schema": output_schema,
        }
        
        return langchain_decorated
    
    # Soporte para uso con y sin paréntesis
    if func is not None:
        return decorator(func)
    return decorator
