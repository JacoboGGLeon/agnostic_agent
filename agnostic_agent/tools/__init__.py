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
from . import basic, math, semantic, finance, introspection, composition_tools

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
_register_module_tools(finance)
_register_module_tools(introspection)
_register_module_tools(composition_tools)

# Re-exportamos funciones clave para compatibilidad hacia atrás si alguien importaba directamente
# (Aunque lo ideal es usar el registro)
to_upper = basic.to_upper
word_count = basic.word_count
is_palindrome = basic.is_palindrome
eval_math_expression = math.eval_math_expression
sum_numbers = math.sum_numbers
average_numbers = math.average_numbers
embed_texts = semantic.embed_texts
semantic_search_in_memory = semantic.semantic_search_in_memory
context_search_in_csv = semantic.context_search_in_csv
embed_context_tables = semantic.embed_context_tables
rerank_docs = semantic.rerank_docs
judge_row_with_context = semantic.judge_row_with_context
search_knowledge_base = semantic.search_knowledge_base
query_transactions_db = finance.query_transactions_db
query_accounting_db = finance.query_accounting_db
finance_sources_status = finance.finance_sources_status
get_saneamiento_rate = finance.get_saneamiento_rate
reconcile_credit_accounting = finance.reconcile_credit_accounting
nl2sql_sqlite = introspection.nl2sql_sqlite
nl2sql_agent_sqlite = introspection.nl2sql_agent_sqlite
knowledge_voyague_nl2sql_agent = introspection.knowledge_voyague_nl2sql_agent
knowledge_voyague_nl2semantic_agent = introspection.knowledge_voyague_nl2semantic_agent
compose_skills = composition_tools.compose_skills


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
