from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


NodeStatus = Literal["ok", "warn", "error"]


class PipelineEvent(BaseModel):
    node: str
    status: NodeStatus = "ok"
    duration_ms: int = 0
    payload: Dict[str, Any] = Field(default_factory=dict)


class UserSection(BaseModel):
    title: str
    items: List[str] = Field(default_factory=list)


class UserViewModelV2(BaseModel):
    final_answer: str = ""
    sections: List[UserSection] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DeepViewModelV2(BaseModel):
    timeline: List[PipelineEvent] = Field(default_factory=list)
    summary: Optional["DeepSummaryV2"] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    raw: Dict[str, Any] = Field(default_factory=dict)


class DevViewModelV2(BaseModel):
    summary: str = ""
    counts: Dict[str, int] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)


class DeepSummaryV2(BaseModel):
    analyzer: Dict[str, Any] = Field(default_factory=dict)
    planner: Dict[str, Any] = Field(default_factory=dict)
    executor: Dict[str, Any] = Field(default_factory=dict)
    catcher: Dict[str, Any] = Field(default_factory=dict)
    summarizer: Dict[str, Any] = Field(default_factory=dict)
    validator: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class PipelineOutputV2(BaseModel):
    user_out: UserViewModelV2
    deep_out: DeepViewModelV2
    dev_out: DevViewModelV2
    schema_version: str = "v2"
    turn_id: Optional[str] = None
