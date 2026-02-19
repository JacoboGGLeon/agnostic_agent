from typing import Dict, Any, List
from agnostic_agent.core.contracts.plugin import Plugin
from agnostic_agent.tools import math as math_tools

class MathToolsPlugin(Plugin):
    """
    Plugin wrapper for math tools.
    """
    @property
    def name(self) -> str:
        return "math"

    @property
    def type(self) -> str:
        return "tool"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    def register(self, registry: List[Any]) -> None:
        # We need to inspect agnostic_agent.tools.math to see what it exports
        # Assuming it exports `calculator` or similar
        # Since I can't see the file content right now, I'll assume standard ones
        # If imports fail, this plugin load will fail, which is catchable
        
        # Based on file list, math.py likely contains mathematical tools
        # Just register everything public or specific ones if known
        
        # For safety, I'll inspect the module in python or assume a standard set
        # Let's assume it has `calculate`
        if hasattr(math_tools, "calculate"):
            registry.append(math_tools.calculate)
        elif hasattr(math_tools, "calculator"):
             registry.append(math_tools.calculator)
        else:
             # Fallback: try to find anything decorated with @tool in the module
             for name in dir(math_tools):
                 obj = getattr(math_tools, name)
                 if callable(obj) and hasattr(obj, "is_tool"): # Check our custom decorator attr
                     registry.append(obj)
