import pytest

from agnostic_agent.app.errors import ConfigurationError
from agnostic_agent.plugins.manager import PluginManager


def test_plugin_manager_raises_in_strict_mode(monkeypatch):
    monkeypatch.setenv("AGNOSTIC_STRICT_PLUGINS", "true")
    config = {
        "tool": {
            "broken": {
                "enabled": True,
                "path": "does_not_exist.py",
            }
        }
    }
    manager = PluginManager(config)
    with pytest.raises(ConfigurationError):
        manager.load_plugins()


def test_plugin_manager_ignores_failures_when_not_strict(monkeypatch):
    monkeypatch.setenv("AGNOSTIC_STRICT_PLUGINS", "false")
    config = {
        "tool": {
            "broken": {
                "enabled": True,
                "path": "does_not_exist.py",
            }
        }
    }
    manager = PluginManager(config)
    manager.load_plugins()
    assert manager.plugins == {}
