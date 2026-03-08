from __future__ import annotations

import uuid
from typing import Any, Dict

from agnostic_agent.protocols.validator import validate_scp_plan
from agnostic_agent.runtime import get_skill_invoker
from agnostic_agent.tools.composition import execute_composition
from agnostic_agent.tools.decorators import tool


def _default_skill_invoker(skill_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "success",
        "outputs": {"skill": skill_name, "inputs": inputs, "note": "default invoker"},
    }


@tool(mode="public")
def compose_skills(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute SCP composition plans (sequential, parallel, conditional, map, tree).
    """
    ok, errors = validate_scp_plan(plan)
    if not ok:
        return {
            "status": "error",
            "op": str(plan.get("op", "")),
            "children": [],
            "outputs": {},
            "errors": [{"code": "SCP_INVALID_PLAN", "message": "; ".join(errors)}],
            "artifacts": [],
        }
    invoker = get_skill_invoker() or _default_skill_invoker
    return execute_composition(
        plan=plan,
        invoke_skill=invoker,
        run_id=f"cmp_{uuid.uuid4().hex[:10]}",
    )
