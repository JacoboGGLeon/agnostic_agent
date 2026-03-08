from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generator

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

    def chat_normalized(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Normalized response contract shared by all providers.
        Providers can override this for richer metadata.
        """
        text = self.chat(messages, **kwargs)
        return {
            "text": text or "",
            "tool_calls": [],
            "usage": {},
            "finish_reason": None,
            "raw": {"text": text or ""},
        }
