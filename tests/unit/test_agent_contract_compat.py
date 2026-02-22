import pytest
from unittest.mock import MagicMock, patch
from agnostic_agent.agent import Agent, AgentInput
from agnostic_agent.capabilities import PlannerConfig

@pytest.fixture
def mock_graph_app():
    mock_app = MagicMock()
    # Mocking invoke response
    mock_app.invoke.return_value = {
        "messages": [],
        "dev_out": "dev output",
        "deep_out": "deep output",
        "user_out": "user output",
        "summary": {},
        "tool_runs": []
    }
    return mock_app

@pytest.fixture
def mock_dependencies():
    with patch('agnostic_agent.agent.build_planner_llm') as mock_llm, \
         patch('agnostic_agent.agent.load_logic') as mock_load_logic, \
         patch('agnostic_agent.agent.get_default_tools') as mock_tools, \
         patch('agnostic_agent.app.turn_service.read_memory') as mock_read_mem, \
         patch('agnostic_agent.app.turn_service.write_memory') as mock_write_mem, \
         patch('agnostic_agent.agent.select_knowledge_bases') as mock_select_kb:
        
        mock_tools.return_value = []
        mock_read_mem.return_value = "memory context"
        mock_select_kb.return_value = []
        
        yield {
            "load_logic": mock_load_logic,
            "read_memory": mock_read_mem,
            "write_memory": mock_write_mem
        }

def test_agent_init(mock_dependencies, mock_graph_app):
    mock_dependencies["load_logic"].return_value = mock_graph_app
    
    agent = Agent.init()
    assert agent is not None
    assert agent.turn_service is not None
    assert agent.graph_app == mock_graph_app

def test_agent_run_turn(mock_dependencies, mock_graph_app):
    mock_dependencies["load_logic"].return_value = mock_graph_app
    
    agent = Agent.init()
    result = agent.run_turn("Hello world")
    
    assert isinstance(result, dict)
    assert "user_out" in result
    assert result["user_out"]["final_answer"] == "user output"
    
    # Verify graph was invoked
    mock_graph_app.invoke.assert_called_once()
    
    # Verify memory was written
    mock_dependencies["write_memory"].assert_called_once()

def test_agent_run_turn_with_dict(mock_dependencies, mock_graph_app):
    mock_dependencies["load_logic"].return_value = mock_graph_app
    
    agent = Agent.init()
    result = agent.run_turn({"user_prompt": "Hello", "session_id": "test_sess"})
    
    assert isinstance(result, dict)
    assert result["user_out"]["final_answer"] == "user output"

def test_agent_backward_compat_attributes(mock_dependencies, mock_graph_app):
    mock_dependencies["load_logic"].return_value = mock_graph_app
    
    agent = Agent.init()
    # Check that old attributes still exist and are accessible
    assert hasattr(agent, "memory_cfg")
    assert hasattr(agent, "knowledge_bases")
    assert hasattr(agent, "setup_config")
