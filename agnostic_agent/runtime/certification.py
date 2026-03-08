from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


MaturityLevel = Literal["L0 Runnable", "L1 Validated", "L2 Certified", "L3 Production"]


class CertificationReport(BaseModel):
    skill_name: str
    level: MaturityLevel
    checks: Dict[str, bool] = Field(default_factory=dict)
    notes: Dict[str, Any] = Field(default_factory=dict)


def assess_skill_maturity(
    *,
    skill_name: str,
    checks: Dict[str, bool],
    notes: Dict[str, Any] | None = None,
) -> CertificationReport:
    """
    Minimal TEP-compatible maturity evaluation.
    """
    manifest_valid = bool(checks.get("manifest_valid"))
    smoke_ok = bool(checks.get("smoke_ok"))
    schema_valid = bool(checks.get("schema_valid"))
    errors_normalized = bool(checks.get("errors_normalized"))
    tool_contracts = bool(checks.get("tool_contracts"))
    knowledge_contracts = bool(checks.get("knowledge_contracts"))
    artifacts_emitted = bool(checks.get("artifacts_emitted"))
    observability_complete = bool(checks.get("observability_complete"))
    version_stable = bool(checks.get("version_stable"))

    level: MaturityLevel = "L0 Runnable"
    if manifest_valid and smoke_ok and schema_valid and errors_normalized:
        level = "L1 Validated"
    if level == "L1 Validated" and tool_contracts and knowledge_contracts and artifacts_emitted:
        level = "L2 Certified"
    if level == "L2 Certified" and observability_complete and version_stable:
        level = "L3 Production"

    return CertificationReport(
        skill_name=skill_name,
        level=level,
        checks=checks,
        notes=notes or {},
    )
