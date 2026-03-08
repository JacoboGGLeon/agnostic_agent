from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol

from agnostic_agent.app.errors import ToolExecutionError
from agnostic_agent.core.contracts.pipeline_tool import (
    PipelineToolError,
    PipelineToolInput,
    PipelineToolOutput,
)


class PipelineTool(Protocol):
    name: str
    contract_version: str

    def invoke(self, payload: PipelineToolInput) -> PipelineToolOutput: ...


@dataclass
class CallablePipelineTool:
    """
    Adapter to execute pipeline stages as contract-first tools.
    """

    name: str
    handler: Callable[[PipelineToolInput], Dict[str, Any]]
    contract_version: str = "pipeline-tool/v1"

    def invoke(self, payload: PipelineToolInput) -> PipelineToolOutput:
        try:
            patch = self.handler(payload)
            if not isinstance(patch, dict):
                raise TypeError(
                    f"Pipeline tool '{self.name}' must return dict state patch, got {type(patch).__name__}."
                )
            return PipelineToolOutput(
                tool_name=self.name,
                contract_version=self.contract_version,
                ok=True,
                state_patch=patch,
                raw={"keys": sorted(patch.keys())},
            )
        except ToolExecutionError as e:
            return PipelineToolOutput(
                tool_name=self.name,
                contract_version=self.contract_version,
                ok=False,
                errors=[
                    PipelineToolError(
                        code=e.code,
                        message=e.message,
                        details=e.details,
                    )
                ],
            )
        except Exception as e:
            return PipelineToolOutput(
                tool_name=self.name,
                contract_version=self.contract_version,
                ok=False,
                errors=[
                    PipelineToolError(
                        code="PIPELINE_TOOL_ERROR",
                        message=f"{self.name} failed: {e}",
                        details={"tool_name": self.name},
                    )
                ],
            )


def invoke_pipeline_tool_or_raise(
    tool: PipelineTool,
    *,
    state: Dict[str, Any],
    context: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    payload = PipelineToolInput(state=state, context=context, metadata=metadata)
    out = tool.invoke(payload)
    if out.ok:
        return out.state_patch
    err = out.errors[0] if out.errors else None
    message = err.message if err else f"Pipeline tool '{tool.name}' failed."
    details = err.details if err else {"tool_name": tool.name}
    raise ToolExecutionError(message=message, tool_name=tool.name, details=details)
