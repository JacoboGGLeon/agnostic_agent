from typing import Optional, Dict, Any

class AgnosticAgentError(Exception):
    """Base class for all agent errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

class ConfigurationError(AgnosticAgentError):
    """Raised when configuration is invalid or missing."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="CONFIG_ERROR", details=details)

class ProviderError(AgnosticAgentError):
    """Raised when a provider (LLM, Embedding, VectorStore) fails."""
    def __init__(self, message: str, provider: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["provider"] = provider
        super().__init__(message, code="PROVIDER_ERROR", details=details)

class TurnExecutionError(AgnosticAgentError):
    """Raised when a turn execution fails."""
    def __init__(self, message: str, step: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["step"] = step
        super().__init__(message, code="TURN_EXECUTION_ERROR", details=details)

class ToolExecutionError(AgnosticAgentError):
    """Raised when a tool execution fails."""
    def __init__(self, message: str, tool_name: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["tool_name"] = tool_name
        super().__init__(message, code="TOOL_EXECUTION_ERROR", details=details)

class PluginError(AgnosticAgentError):
    """Raised when a plugin fails to load or execute."""
    def __init__(self, message: str, plugin_name: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["plugin_name"] = plugin_name
        super().__init__(message, code="PLUGIN_ERROR", details=details)
