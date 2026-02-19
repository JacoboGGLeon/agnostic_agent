from typing import Dict, Any, List
from agnostic_agent.core.contracts.plugin import Plugin
# We import from original location to reuse logic without copy-paste
from agnostic_agent.tools import basic 

class BasicToolsPlugin(Plugin):
    """
    Plugin wrapper for basic text processing tools.
    """
    @property
    def name(self) -> str:
        return "basic"

    @property
    def type(self) -> str:
        return "tool"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    def register(self, registry: List[Any]) -> None:
        # Assuming registry is a list where we append tools
        registry.extend([
            basic.to_upper,
            basic.word_count,
            basic.is_palindrome
        ])
