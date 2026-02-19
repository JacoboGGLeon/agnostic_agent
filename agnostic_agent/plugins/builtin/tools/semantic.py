from typing import Dict, Any, List
from agnostic_agent.core.contracts.plugin import Plugin
from agnostic_agent.tools import semantic

class SemanticToolsPlugin(Plugin):
    """
    Plugin wrapper for semantic search tools.
    """
    @property
    def name(self) -> str:
        return "semantic"

    @property
    def type(self) -> str:
        return "tool"
    
    def initialize(self, config: Dict[str, Any]) -> None:
         # Config might contain keys etc, but tools usually configured globally or via instance
         pass
    
    def register(self, registry: List[Any]) -> None:
         # Register semantic search tools
         # Based on typical implementation
         if hasattr(semantic, "semantic_search"):
             registry.append(semantic.semantic_search)
         if hasattr(semantic, "read_semantics"): # Example name
             registry.append(semantic.read_semantics)
         
         # Fallback discovery
         for name in dir(semantic):
             obj = getattr(semantic, name)
             if callable(obj) and hasattr(obj, "is_tool"):
                 registry.append(obj)
