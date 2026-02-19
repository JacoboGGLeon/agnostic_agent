from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class Plugin(ABC):
    """
    Abstract base class for all plugins.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the plugin."""
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        """
        Type of plugin: 'tool', 'skill', 'memory', 'ui.panel'.
        """
        pass
        
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
        
    @abstractmethod
    def register(self, registry: Any) -> None:
        """
        Register plugin components into the system registry.
        The registry type depends on the plugin type.
        """
        pass

    def render(self, context: Any = None) -> None:
        """
        Render the UI component. 
        Only used if type starts with 'ui.'.
        Context is usually the Streamlit module or a container.
        """
        pass
