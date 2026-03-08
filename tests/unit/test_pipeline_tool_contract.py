from agnostic_agent.app.errors import ToolExecutionError
from agnostic_agent.core.contracts.pipeline_tool import PipelineToolInput
from agnostic_agent.tools.pipeline_runtime import (
    CallablePipelineTool,
    invoke_pipeline_tool_or_raise,
)


def test_callable_pipeline_tool_success():
    tool = CallablePipelineTool(
        name="pipeline.test",
        handler=lambda payload: {"messages": [], "executor_steps": []},
    )
    out = tool.invoke(PipelineToolInput(state={"messages": []}, context={}, metadata={}))
    assert out.ok is True
    assert out.state_patch["executor_steps"] == []
    assert out.contract_version == "pipeline-tool/v1"


def test_callable_pipeline_tool_error_output():
    def _boom(payload):
        raise RuntimeError("fail")

    tool = CallablePipelineTool(name="pipeline.test", handler=_boom)
    out = tool.invoke(PipelineToolInput(state={}, context={}, metadata={}))
    assert out.ok is False
    assert out.errors
    assert out.errors[0].code == "PIPELINE_TOOL_ERROR"


def test_invoke_pipeline_tool_or_raise_raises_tool_execution_error():
    def _boom(payload):
        raise RuntimeError("kaboom")

    tool = CallablePipelineTool(name="pipeline.test", handler=_boom)
    try:
        invoke_pipeline_tool_or_raise(
            tool,
            state={"messages": []},
            context={},
            metadata={"node": "test"},
        )
        assert False, "expected ToolExecutionError"
    except ToolExecutionError as e:
        assert e.code == "TOOL_EXECUTION_ERROR"
        assert e.details.get("tool_name") == "pipeline.test"
