from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

from pydantic import BaseModel, Field


TestMode = Literal["explicit", "auto", "explicit_or_auto"]
ComponentType = Literal["skill", "tool", "knowledge_adapter"]


class TEPRecord(BaseModel):
    component_type: ComponentType
    component_name: str
    check_id: str
    passed: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class TEPBundle(BaseModel):
    protocol: str = "test-evidence/v1"
    mode: TestMode = "explicit_or_auto"
    records: List[TEPRecord] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


def validate_tep_minimum_checks(bundle: TEPBundle) -> Tuple[bool, list[str]]:
    """
    Minimum checks aligned with technical report section 8.5.
    """
    required_by_component: Dict[str, set[str]] = {
        "tool": {
            "importability",
            "schema_conformance",
            "null_empty_handling",
            "timeout_enforcement",
            "failure_shape_normalization",
        },
        "knowledge_adapter": {
            "importability",
            "search_shape",
            "get_shape",
            "provenance_presence",
            "empty_result_behavior",
        },
        "skill": {
            "manifest_validation",
            "smoke_execution",
            "input_schema_validation",
            "output_schema_validation",
            "error_normalization",
        },
    }

    missing: list[str] = []
    by_component: Dict[tuple[str, str], set[str]] = {}

    for record in bundle.records:
        key = (record.component_type, record.component_name)
        if key not in by_component:
            by_component[key] = set()
        if record.passed:
            by_component[key].add(record.check_id)

    for (component_type, component_name), checks in by_component.items():
        required = required_by_component.get(component_type, set())
        for check in sorted(required - checks):
            missing.append(f"{component_type}:{component_name} missing passed check {check}")

    return len(missing) == 0, missing
