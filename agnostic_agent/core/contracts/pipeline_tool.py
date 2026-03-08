from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineToolInput(BaseModel):
    """
    Canonical input envelope for pipeline-internal tools.
    """

    state: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineToolError(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class PipelineToolOutput(BaseModel):
    """
    Canonical output envelope for pipeline-internal tools.
    """

    tool_name: str
    contract_version: str = "pipeline-tool/v1"
    ok: bool = True
    state_patch: Dict[str, Any] = Field(default_factory=dict)
    errors: List[PipelineToolError] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None
