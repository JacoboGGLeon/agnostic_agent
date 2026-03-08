from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


CompositionOp = Literal["sequential", "parallel", "conditional", "map", "tree"]


class CompositionStep(BaseModel):
    skill: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    inputs_from: Optional[str] = None


class CompositionPlan(BaseModel):
    op: CompositionOp
    steps: List[CompositionStep] = Field(default_factory=list)
    condition: Optional[Dict[str, Any]] = None
    then_step: Optional[CompositionStep] = None
    else_step: Optional[CompositionStep] = None
    map_items: List[Any] = Field(default_factory=list)
    map_step: Optional[CompositionStep] = None
    root: Optional[CompositionStep] = None
    children: List[CompositionStep] = Field(default_factory=list)
