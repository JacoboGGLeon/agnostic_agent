from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal["respond", "invoke_skill", "invoke_tool", "compose", "fail"]


class SkillManifest(BaseModel):
    api_version: str
    kind: str
    name: str
    version: str
    entrypoint: str
    instructions: str
    input_schema: str
    output_schema: str
    tools: Dict[str, Any] = Field(default_factory=dict)
    knowledge: Dict[str, Any] = Field(default_factory=dict)
    composition: Dict[str, Any] = Field(default_factory=dict)
    testing: Dict[str, Any] = Field(default_factory=dict)
    execution: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunContext(BaseModel):
    run_id: str
    session_id: str = "default"
    caller: str = "runtime"
    goal: str = ""
    inputs: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    state: Dict[str, Any] = Field(default_factory=dict)
    timestamps: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    type: ActionType
    payload: Dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    artifact_id: str
    run_id: str
    kind: str
    producer: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SkillRequest(BaseModel):
    skill_name: str
    skill_version: Optional[str] = None
    goal: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)


class SkillResult(BaseModel):
    status: str = "success"
    outputs: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    children: List[Dict[str, Any]] = Field(default_factory=list)


class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    caller: str = "runtime"
    timeout_s: Optional[float] = None


class ToolResult(BaseModel):
    ok: bool
    data: Any = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeItem(BaseModel):
    id: str
    type: str
    content: Any
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    text: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    structured: Optional[Dict[str, Any]] = None
    usage: Dict[str, Any] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    raw: Any = None
