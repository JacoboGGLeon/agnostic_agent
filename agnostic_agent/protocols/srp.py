from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SkillRuntimeRef(BaseModel):
    name: str
    version: Optional[str] = None


class SkillRuntimeRequest(BaseModel):
    protocol: str = "skill-runtime/v1"
    run_id: str
    skill: SkillRuntimeRef
    goal: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)


class SkillRuntimeError(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class SkillRuntimeResponse(BaseModel):
    protocol: str = "skill-runtime/v1"
    status: Literal["success", "error"] = "success"
    outputs: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[SkillRuntimeError] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    children: List[Dict[str, Any]] = Field(default_factory=list)
