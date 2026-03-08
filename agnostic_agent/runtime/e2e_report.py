from __future__ import annotations

from typing import Any, Dict, List

from agnostic_agent.protocols.validator import validate_srp_response


def build_end_to_end_report(
    *,
    run_id: str,
    prompt_text: str,
    tool_runs: List[Dict[str, Any]],
    protocol_checks: Dict[str, Dict[str, Any]],
    user_answer: str,
) -> Dict[str, Any]:
    srp_ok, srp_errors = validate_srp_response(
        {
            "status": "success",
            "outputs": {"final_answer": user_answer},
            "artifacts": [],
            "errors": [],
            "metrics": {},
            "children": [],
        }
    )
    checks = dict(protocol_checks)
    checks["srp_output_shape"] = {"ok": srp_ok, "errors": srp_errors}

    return {
        "run_id": run_id,
        "prompt_preview": (prompt_text or "")[:140],
        "tool_runs_count": len(tool_runs),
        "protocol_checks": checks,
        "final_answer_non_empty": bool((user_answer or "").strip()),
    }
