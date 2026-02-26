import pytest

from agnostic_agent.graph.state_contracts import validate_node_input, validate_node_output


def test_validate_node_input_requires_messages_for_analyzer():
    with pytest.raises(ValueError):
        validate_node_input("analyzer", {})


def test_validate_node_output_rejects_unknown_keys():
    with pytest.raises(ValueError):
        validate_node_output("executor", {"messages": [], "unexpected": 1})
