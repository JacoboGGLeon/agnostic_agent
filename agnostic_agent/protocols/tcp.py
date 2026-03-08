from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


ToolSideEffects = Literal[
    "read_only",
    "filesystem_write",
    "network_io",
    "external_state",
]

TestMode = Literal["explicit", "auto", "explicit_or_auto"]


class ToolTestingConfig(BaseModel):
    mode: TestMode = "explicit_or_auto"


class ToolContract(BaseModel):
    name: str
    description: str
    input_schema: str
    output_schema: str
    side_effects: ToolSideEffects = "read_only"
    timeout_s: float = Field(default=30.0, gt=0)
    testing: ToolTestingConfig = Field(default_factory=ToolTestingConfig)


class ToolNormalizedResult(BaseModel):
    ok: bool
    data: Any = None
    error: Dict[str, Any] | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
