from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


EventKind = Literal[
    "run.started",
    "skill.invoked",
    "tool.called",
    "knowledge.accessed",
    "skill.completed",
    "run.completed",
    "run.failed",
]


class ArtifactEvent(BaseModel):
    event_id: str
    run_id: str
    kind: EventKind
    producer: str
    timestamp_utc: str
    payload: Dict[str, Any] = Field(default_factory=dict)


def build_event(
    *,
    run_id: str,
    kind: EventKind,
    producer: str,
    payload: Dict[str, Any] | None = None,
) -> ArtifactEvent:
    return ArtifactEvent(
        event_id=f"evt_{uuid4().hex[:12]}",
        run_id=run_id,
        kind=kind,
        producer=producer,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        payload=payload or {},
    )


class ArtifactEmitter:
    def __init__(self) -> None:
        self._events: List[ArtifactEvent] = []

    def emit(
        self,
        *,
        run_id: str,
        kind: EventKind,
        producer: str,
        payload: Dict[str, Any] | None = None,
    ) -> ArtifactEvent:
        evt = build_event(run_id=run_id, kind=kind, producer=producer, payload=payload)
        self._events.append(evt)
        return evt

    def list_events(self) -> List[ArtifactEvent]:
        return list(self._events)
