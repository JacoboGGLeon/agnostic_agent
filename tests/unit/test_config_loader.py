import os
import pytest
from pathlib import Path
from agnostic_agent.config.loader import load_config, AppConfig, settings
from agnostic_agent.config.overrides import override_settings

# Mocks for paths
@pytest.fixture
def mock_config_dir(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    profiles_dir = config_dir / "profiles"
    profiles_dir.mkdir()

    # Base config
    with open(config_dir / "config.yaml", "w") as f:
        f.write("environment: base\ndebug: false\nllm:\n  provider: base_llm")

    # Dev profile
    with open(profiles_dir / "dev.yaml", "w") as f:
        f.write("environment: dev\ndebug: true")

    return config_dir

def test_load_base_config(mock_config_dir):
    config = load_config(config_dir=mock_config_dir)
    assert config.environment == "base"
    assert config.debug is False
    assert config.llm.provider == "base_llm"

def test_load_profile_config(mock_config_dir):
    config = load_config(profile="dev", config_dir=mock_config_dir)
    assert config.environment == "dev"
    assert config.debug is True
    # Inherited from base
    assert config.llm.provider == "base_llm"

def test_env_var_override(mock_config_dir):
    with pytest.MonkeyPatch.context() as m:
        m.setenv("AGNOSTIC_DEBUG", "true")
        m.setenv("AGNOSTIC_LLM_PROVIDER", "env_llm")
        
        config = load_config(config_dir=mock_config_dir)
        
        assert config.debug is True
        assert config.llm.provider == "env_llm"

def test_override_settings_context_manager():
    # Store original to verify restoration
    original_debug = settings.debug
    
    with override_settings({"debug": not original_debug}):
        assert settings.debug != original_debug
    
    assert settings.debug == original_debug

def test_find_config_dir():
    # This is harder to test without mocking the filesystem structure deeply
    # Skipping for now or would need to mock Path.exists and os.getcwd
    pass
