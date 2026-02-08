from __future__ import annotations

"""
Catálogo de herramientas (tools) para el Agnostic Deep Agent 2026.
REGISTRY REFACTORIZADO (v2).

Ahora carga herramientas desde la carpeta `tools/` de forma modular.
Mantiene la interfaz original para compatibilidad.
"""

from typing import List, Dict, Any
import logging
from langchain_core.tools import Tool, BaseTool

# Logger
logger = logging.getLogger(__name__)

# Importamos los módulos de tools explícitamente para llenar el registro
# Usamos imports relativos para evitar circularidad si se importara agnostic_agent.tools antes
from . import basic, math, semantic

# Construimos el registro global agregando las tools de cada submódulo
TOOL_REGISTRY: Dict[str, Any] = {}

def _register_module_tools(module):
    """Scan module for Tool objects and register them."""
    for name in dir(module):
        obj = getattr(module, name)
        # Check if it's a langchain tool (function decorated with @tool generates BaseTool subclass instance)
        if isinstance(obj, BaseTool):
             TOOL_REGISTRY[obj.name] = obj

# Registramos explícitamente los módulos
_register_module_tools(basic)
_register_module_tools(math)
_register_module_tools(semantic)

# Re-exportamos funciones clave para compatibilidad hacia atrás si alguien importaba directamente
# (Aunque lo ideal es usar el registro)
to_upper = basic.to_upper
word_count = basic.word_count
is_palindrome = basic.is_palindrome
eval_math_expression = math.eval_math_expression
sum_numbers = math.sum_numbers
average_numbers = math.average_numbers
embed_texts = semantic.embed_texts
semantic_search = semantic.semantic_search
semantic_search_in_csv = semantic.semantic_search_in_csv
embed_context_tables = semantic.embed_context_tables
rerank_qwen3 = semantic.rerank_qwen3
judge_row_with_context = semantic.judge_row_with_context
search_knowledge_base = semantic.search_knowledge_base


def get_default_tools(enabled_names: List[str] | None = None) -> List[Any]:
    """
    Tools por defecto del agente agnóstico.

    - Si enabled_names es None → devuelve TODAS las tools registradas.
    - Si enabled_names es una lista → sólo devuelve las que estén en TOOL_REGISTRY.
    """
    if enabled_names is None:
        return list(TOOL_REGISTRY.values())
    return [TOOL_REGISTRY[name] for name in enabled_names if name in TOOL_REGISTRY]


def get_tools_by_names(names: List[str]) -> List[Any]:
    """
    Permite seleccionar tools por nombre desde TOOL_REGISTRY.
    Equivalente a get_default_tools(names), se mantiene por claridad semántica.
    """
    return get_default_tools(names)
