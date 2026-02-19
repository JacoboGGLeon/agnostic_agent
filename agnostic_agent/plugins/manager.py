import importlib
import logging
from typing import Dict, Any, List, Type
from pathlib import Path
from agnostic_agent.core.contracts.plugin import Plugin
from agnostic_agent.app.errors import ConfigurationError

logger = logging.getLogger(__name__)

class PluginManager:
    """
    Manages discovery, loading, and registration of plugins.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plugins: Dict[str, Plugin] = {}
        
    def load_plugins(self):
        """Load all enabled plugins from configuration."""
        # Config structure:
        # plugins:
        #   tool:
        #     name: {enabled: true, path: "..."}
        
        for category, plugins_conf in self.config.items():
            if not isinstance(plugins_conf, dict): continue
            
            for name, conf in plugins_conf.items():
                if not conf.get("enabled", True):
                    continue
                
                path = conf.get("path")
                try:
                    plugin_instance = self._load_plugin(name, category, path, conf.get("config", {}))
                    # Store with category key "tool.my_tool" or "ui.panel.my_panel"
                    self.plugins[f"{category}.{name}"] = plugin_instance
                    logger.info(f"Loaded plugin: {category}.{name}")
                except Exception as e:
                    logger.error(f"Failed to load plugin {name}: {e}")
                    
    def _load_plugin(self, name: str, category: str, path: str, config: Dict[str, Any]) -> Plugin:
        module = None
        
        if path and (path.endswith(".py") or Path(path).exists()):
            spec = importlib.util.spec_from_file_location(f"plugin_{name}", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        else:
            # Fallback path logic
            module_path = path
            if not module_path:
                # Default locations
                if category.startswith("ui."):
                     module_path = f"agnostic_agent.plugins.builtin.ui_panels.{name}"
                else:
                     module_path = f"agnostic_agent.plugins.builtin.{category}s.{name}" if not category.endswith("s") else f"agnostic_agent.plugins.builtin.{category}.{name}"
                     
                     # Try singular/plural heuristic if not found?
                     # Simplified: Just rely on user config or standard `builtin.tools.basic`
                     # Actually existing code assumed `builtin.{category}.{name}`.
                     # Let's keep it simple and flexible.
                     module_path = f"agnostic_agent.plugins.builtin.{category}.{name}"
            
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                 # Try adding 's' to category for builtins (tool -> tools)
                 if not path and not module_path.endswith("s"): # Heuristic
                     try:
                         module_path_s = module_path.replace(f".{category}.", f".{category}s.")
                         module = importlib.import_module(module_path_s)
                     except ImportError:
                         raise ConfigurationError(f"Could not import plugin module: {module_path}")
                 else:
                     raise ConfigurationError(f"Could not import plugin module: {module_path}")

        # Find Plugin subclass
        plugin_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Plugin) and attr is not Plugin:
                plugin_class = attr
                break
        
        if not plugin_class:
            raise ConfigurationError(f"No Plugin subclass found in {path or module_path}")
            
        instance = plugin_class()
        instance.initialize(config)
        return instance

    def register_all(self, registry_map: Dict[str, Any]):
        """
        Register all loaded plugins into their respective registries.
        registry_map: {"tool": tool_registry, ...}
        """
        for plugin_key, plugin in self.plugins.items():
            # plugin.type might be 'ui.panel', 'tool', etc.
            # Registry map keys should match
            registry = registry_map.get(plugin.type)
            if registry:
                plugin.register(registry)

    def get_ui_plugins(self, location: str = None) -> List[Plugin]:
        """
        Return list of UI plugins. 
        If location is provided, filters by type 'ui.{location}'.
        e.g. location='tab' -> type='ui.tab'
        """
        res = []
        for p in self.plugins.values():
            if not p.type.startswith("ui."):
                continue
            if location:
                if p.type == f"ui.{location}":
                    res.append(p)
            else:
                res.append(p)
        return res
