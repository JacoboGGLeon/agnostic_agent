import warnings
from agnostic_agent.agent import Agent
from agnostic_agent.core.models.io_models import AgentInput, AgentOutput
from agnostic_agent.knowledge import KnowledgeBase, get_default_context
from agnostic_agent.tools import get_default_tools

# Deprecated classes/functions mapping
class AgentSession(Agent):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "AgentSession is deprecated and will be removed in v3. Use Agent instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

def get_legacy_context():
    warnings.warn(
        "get_legacy_context is deprecated. Use get_default_context or KnowledgeBase directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_default_context()
