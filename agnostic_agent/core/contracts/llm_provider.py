from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generator, Union

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream a response from the LLM."""
        pass
    
    # Ideally we should support chat messages format too
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate response from a list of messages."""
        pass
