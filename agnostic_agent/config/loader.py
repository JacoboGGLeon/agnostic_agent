import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union
from .schema import AppConfig

def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def find_config_dir() -> Path:
    """Find the configuration directory."""
    # Try generic locations: env var, current dir, parent dirs
    if os.getenv("AGNOSTIC_CONFIG_DIR"):
        return Path(os.getenv("AGNOSTIC_CONFIG_DIR"))
    
    # Check if we are running from source user repo
    current = Path.cwd()
    if (current / "config" / "config.yaml").exists():
        return current / "config"
    
    # Fallback to package-relative (up 2 levels from here)
    # agnostic_agent/config/loader.py -> agnostic_agent/ -> parent -> config
    package_config = Path(__file__).parent.parent.parent / "config"
    if package_config.exists():
        return package_config

    return Path("config")

def load_config(profile: Optional[str] = None, config_dir: Optional[Union[str, Path]] = None) -> AppConfig:
    """
    Load configuration with precedence:
    1. Base config.yaml
    2. Profile config (e.g. profiles/dev.yaml)
    3. Environment variables (AGNOSTIC_*)
    """
    if config_dir:
        base_dir = Path(config_dir)
    else:
        base_dir = find_config_dir()

    # 1. Base Config
    config_data = load_yaml_config(base_dir / "config.yaml")

    # 2. Profile Config
    env_profile = os.getenv("AGNOSTIC_PROFILE")
    active_profile = profile if profile is not None else env_profile
    
    if active_profile:
        profile_path = base_dir / "profiles" / f"{active_profile}.yaml"
        if profile_path.exists():
            profile_data = load_yaml_config(profile_path)
            config_data = _merge_dicts(config_data, profile_data)

    # 3. Env Vars (Simple flat override for now, can be expanded)
    # Allows generic override like AGNOSTIC_DEBUG=true
    if os.getenv("AGNOSTIC_DEBUG"):
        config_data["debug"] = os.getenv("AGNOSTIC_DEBUG", "").lower() == "true"
    
    # Provider overrides via env
    if os.getenv("AGNOSTIC_LLM_PROVIDER"):
        if "llm" not in config_data: config_data["llm"] = {}
        config_data["llm"]["provider"] = os.getenv("AGNOSTIC_LLM_PROVIDER")

    return AppConfig(**config_data)

# Singleton instance
settings = load_config()
