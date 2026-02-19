from contextlib import contextmanager
from typing import Dict, Any, Generator
from .loader import settings, AppConfig, _merge_dicts
import copy

@contextmanager
def override_settings(overrides: Dict[str, Any]) -> Generator[AppConfig, None, None]:
    """
    Context manager to temporarily override settings.
    Useful for testing or specific runtime scenarios.
    """
    original_settings = copy.deepcopy(settings)
    
    # Hacky way to update the pydantic model in place or replace it
    # Pydantic models are immutable by default if configured so, but here we just replace the singleton content
    # A better way is to create a new model.
    
    current_dict = settings.model_dump()
    new_dict = _merge_dicts(current_dict, overrides)
    new_settings = AppConfig(**new_dict)
    
    # Update the global settings object attributes
    # This is slightly dangerous in threaded apps but fine for this CLI/Streamlit scope
    for key, value in new_settings.model_dump().items():
        setattr(settings, key, getattr(new_settings, key))
        
    try:
        yield settings
    finally:
        # Restore
        for key, value in original_settings.model_dump().items():
            setattr(settings, key, getattr(original_settings, key))
