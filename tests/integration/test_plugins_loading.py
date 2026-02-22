import pytest
import os
from agnostic_agent.plugins.manager import PluginManager

@pytest.fixture
def plugin_config(tmp_path):
    # Create a dummy plugin file
    plugin_file = tmp_path / "test_plugin.py"
    plugin_file.write_text("""
from agnostic_agent.core.contracts.plugin import Plugin
class TestPlugin(Plugin):
    @property
    def name(self): return "test"
    @property
    def type(self): return "tool"
    def initialize(self, cfg): pass
    def register(self, reg): reg.append("registered_tool")
""")
    
    return {
        "tool": {
            "test_plugin": {
                "enabled": True,
                "path": str(plugin_file),
                "config": {"foo": "bar"}
            },
            "basic": {
                "enabled": True,
                "path": None # Builtin
            }
        }
    }

def test_load_plugins(plugin_config):
    manager = PluginManager(plugin_config)
    manager.load_plugins()
    
    assert "tool.test_plugin" in manager.plugins
    assert "tool.basic" in manager.plugins
    
    # Test registration
    registry = []
    manager.register_all({"tool": registry})
    
    assert "registered_tool" in registry
    # Basic plugin registers 3 tools
    assert len(registry) >= 4
