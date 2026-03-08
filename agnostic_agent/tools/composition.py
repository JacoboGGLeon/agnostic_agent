from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

from agnostic_agent.protocols.scp import CompositionPlan
from agnostic_agent.runtime.artifacts import ArtifactEmitter


def _safe_invoke(
    *,
    emitter: ArtifactEmitter,
    run_id: str,
    idx: int,
    skill: str,
    invoke_skill: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    emitter.emit(
        run_id=run_id,
        kind="skill.invoked",
        producer="composition_tool",
        payload={"index": idx, "skill": skill},
    )
    try:
        result = invoke_skill(skill, inputs)
        emitter.emit(
            run_id=run_id,
            kind="skill.completed",
            producer="composition_tool",
            payload={"index": idx, "skill": skill},
        )
        return result
    except Exception as e:
        emitter.emit(
            run_id=run_id,
            kind="run.failed",
            producer="composition_tool",
            payload={"index": idx, "skill": skill, "error": str(e)},
        )
        return {"status": "error", "outputs": {}, "errors": [{"code": "COMPOSITION_SKILL_ERROR", "message": str(e)}]}


def _eval_condition(condition: Dict[str, Any]) -> bool:
    if not isinstance(condition, dict):
        return False
    if "value" in condition:
        return bool(condition.get("value"))
    equals = condition.get("equals")
    if isinstance(equals, dict):
        return equals.get("left") == equals.get("right")
    return False


def execute_composition(
    *,
    plan: Dict[str, Any],
    invoke_skill: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    run_id: str,
    max_workers: int = 4,
) -> Dict[str, Any]:
    parsed = CompositionPlan(**plan)
    emitter = ArtifactEmitter()
    emitter.emit(run_id=run_id, kind="run.started", producer="composition_tool", payload={"op": parsed.op})

    if parsed.op == "sequential":
        children: List[Dict[str, Any]] = []
        last_output: Dict[str, Any] = {}
        for idx, step in enumerate(parsed.steps):
            inputs = dict(step.inputs)
            if step.inputs_from and step.inputs_from == "prev.outputs":
                inputs.update(last_output)
            result = _safe_invoke(
                emitter=emitter,
                run_id=run_id,
                idx=idx,
                skill=step.skill,
                invoke_skill=invoke_skill,
                inputs=inputs,
            )
            last_output = result.get("outputs", {}) if isinstance(result, dict) else {}
            children.append({"index": idx, "skill": step.skill, "result": result})
        emitter.emit(
            run_id=run_id,
            kind="run.completed",
            producer="composition_tool",
            payload={"children": len(children)},
        )
        return {
            "status": "success",
            "op": parsed.op,
            "children": children,
            "outputs": children[-1]["result"].get("outputs", {}) if children else {},
            "artifacts": [e.model_dump() for e in emitter.list_events()],
        }

    if parsed.op == "parallel":
        children: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_map = {}
            for idx, step in enumerate(parsed.steps):
                emitter.emit(
                    run_id=run_id,
                    kind="skill.invoked",
                    producer="composition_tool",
                    payload={"index": idx, "skill": step.skill},
                )
                fut = pool.submit(_safe_invoke, emitter=emitter, run_id=run_id, idx=idx, skill=step.skill, invoke_skill=invoke_skill, inputs=dict(step.inputs))
                fut_map[fut] = (idx, step.skill)

            for fut in as_completed(fut_map):
                idx, skill = fut_map[fut]
                result = fut.result()
                children.append({"index": idx, "skill": skill, "result": result})
        children = sorted(children, key=lambda x: x["index"])
        merged_outputs: Dict[str, Any] = {}
        for child in children:
            result = child.get("result", {})
            if isinstance(result, dict) and isinstance(result.get("outputs"), dict):
                merged_outputs[child["skill"]] = result["outputs"]

        emitter.emit(
            run_id=run_id,
            kind="run.completed",
            producer="composition_tool",
            payload={"children": len(children)},
        )
        return {
            "status": "success",
            "op": parsed.op,
            "children": children,
            "outputs": merged_outputs,
            "artifacts": [e.model_dump() for e in emitter.list_events()],
        }

    if parsed.op == "conditional":
        if parsed.then_step is None or parsed.else_step is None:
            return {
                "status": "error",
                "op": parsed.op,
                "children": [],
                "outputs": {},
                "errors": [{"code": "SCP_INVALID_PLAN", "message": "conditional requires then_step and else_step"}],
                "artifacts": [e.model_dump() for e in emitter.list_events()],
            }
        branch = parsed.then_step if _eval_condition(parsed.condition or {}) else parsed.else_step
        result = _safe_invoke(
            emitter=emitter,
            run_id=run_id,
            idx=0,
            skill=branch.skill,
            invoke_skill=invoke_skill,
            inputs=dict(branch.inputs),
        )
        emitter.emit(
            run_id=run_id,
            kind="run.completed",
            producer="composition_tool",
            payload={"children": 1, "branch": branch.skill},
        )
        return {
            "status": "success",
            "op": parsed.op,
            "children": [{"index": 0, "skill": branch.skill, "result": result}],
            "outputs": result.get("outputs", {}) if isinstance(result, dict) else {},
            "artifacts": [e.model_dump() for e in emitter.list_events()],
        }

    if parsed.op == "map":
        if parsed.map_step is None:
            return {
                "status": "error",
                "op": parsed.op,
                "children": [],
                "outputs": {},
                "errors": [{"code": "SCP_INVALID_PLAN", "message": "map requires map_step"}],
                "artifacts": [e.model_dump() for e in emitter.list_events()],
            }
        children = []
        for idx, item in enumerate(parsed.map_items):
            step_inputs = dict(parsed.map_step.inputs)
            step_inputs["item"] = item
            result = _safe_invoke(
                emitter=emitter,
                run_id=run_id,
                idx=idx,
                skill=parsed.map_step.skill,
                invoke_skill=invoke_skill,
                inputs=step_inputs,
            )
            children.append({"index": idx, "skill": parsed.map_step.skill, "item": item, "result": result})
        emitter.emit(run_id=run_id, kind="run.completed", producer="composition_tool", payload={"children": len(children)})
        return {
            "status": "success",
            "op": parsed.op,
            "children": children,
            "outputs": [c["result"].get("outputs", {}) for c in children],
            "artifacts": [e.model_dump() for e in emitter.list_events()],
        }

    if parsed.op == "tree":
        if parsed.root is None:
            return {
                "status": "error",
                "op": parsed.op,
                "children": [],
                "outputs": {},
                "errors": [{"code": "SCP_INVALID_PLAN", "message": "tree requires root"}],
                "artifacts": [e.model_dump() for e in emitter.list_events()],
            }
        root_result = _safe_invoke(
            emitter=emitter,
            run_id=run_id,
            idx=0,
            skill=parsed.root.skill,
            invoke_skill=invoke_skill,
            inputs=dict(parsed.root.inputs),
        )
        children = [{"index": 0, "skill": parsed.root.skill, "result": root_result, "parent_index": None}]
        base = root_result.get("outputs", {}) if isinstance(root_result, dict) else {}
        for idx, step in enumerate(parsed.children, start=1):
            inputs = dict(step.inputs)
            if step.inputs_from == "root.outputs" and isinstance(base, dict):
                inputs.update(base)
            result = _safe_invoke(
                emitter=emitter,
                run_id=run_id,
                idx=idx,
                skill=step.skill,
                invoke_skill=invoke_skill,
                inputs=inputs,
            )
            children.append({"index": idx, "skill": step.skill, "result": result, "parent_index": 0})
        emitter.emit(run_id=run_id, kind="run.completed", producer="composition_tool", payload={"children": len(children)})
        return {
            "status": "success",
            "op": parsed.op,
            "children": children,
            "outputs": {"root": base},
            "artifacts": [e.model_dump() for e in emitter.list_events()],
        }

    return {
        "status": "error",
        "op": parsed.op,
        "children": [],
        "outputs": {},
        "errors": [{"code": "SCP_NOT_IMPLEMENTED", "message": f"op '{parsed.op}' not implemented yet"}],
        "artifacts": [e.model_dump() for e in emitter.list_events()],
    }
