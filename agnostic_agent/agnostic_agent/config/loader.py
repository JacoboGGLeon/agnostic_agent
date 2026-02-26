"""
Compatibility wrapper.

Canonical config implementation lives in `agnostic_agent.config.loader`.
This module is kept only to avoid breaking old imports.
"""

from agnostic_agent.config.loader import (  # noqa: F401
    _merge_dicts,
    find_config_dir,
    load_config,
    load_yaml_config,
    settings,
)
