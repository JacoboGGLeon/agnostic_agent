from __future__ import annotations

from typing import Any, Callable, Dict

from agnostic_agent.protocols.srp import SkillRuntimeRequest, SkillRuntimeResponse


def invoke_skill_srp(
    *,
    request_payload: Dict[str, Any],
    invoke_skill_impl: Callable[[str, Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validates SRP request/response boundaries for skill invocation.
    """
    req = SkillRuntimeRequest(**request_payload)
    raw = invoke_skill_impl(req.skill.name, req.inputs)
    if not isinstance(raw, dict):
        raw = {"status": "error", "errors": [{"code": "INVALID_SKILL_RESULT", "message": "skill returned non-dict"}]}
    rsp = SkillRuntimeResponse(
        status=raw.get("status", "success"),
        outputs=raw.get("outputs", {}) or {},
        artifacts=raw.get("artifacts", []) or [],
        errors=raw.get("errors", []) or [],
        metrics=raw.get("metrics", {}) or {},
        children=raw.get("children", []) or [],
    )
    return rsp.model_dump()
