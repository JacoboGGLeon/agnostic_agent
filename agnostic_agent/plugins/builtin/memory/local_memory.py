from typing import Dict, Any, List, Optional
import datetime
from agnostic_agent.core.contracts.plugin import Plugin

class LocalMemoryPlugin(Plugin):
    """
    Plugin for local in-memory storage (dictionary-based).
    """
    
    def __init__(self):
        self._db: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "local_memory"

    @property
    def type(self) -> str:
        return "memory"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        pass
    
    def register(self, registry: Any) -> None:
        # registry is expected to be a MemoryRegistry or similar
        # For now, we assume registry has set_provider(self)
        if hasattr(registry, "set_provider"):
            registry.set_provider(self)

    # API
    def read(self, session_id: str) -> Dict[str, Any]:
        return self._db.get(session_id, {})

    def write(self, session_id: str, data: Dict[str, Any]) -> None:
        current = self._db.get(session_id, {})
        current.update(data)
        self._db[session_id] = current
    
    def clear(self, session_id: str) -> None:
        if session_id in self._db:
            del self._db[session_id]
