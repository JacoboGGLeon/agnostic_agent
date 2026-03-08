from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

from agnostic_agent.protocols.scp import CompositionPlan
from agnostic_agent.runtime.artifacts import ArtifactEmitter


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
            emitter.emit(
                run_id=run_id,
                kind="skill.invoked",
                producer="composition_tool",
                payload={"index": idx, "skill": step.skill},
            )
            result = invoke_skill(step.skill, inputs)
            last_output = result.get("outputs", {}) if isinstance(result, dict) else {}
            children.append({"index": idx, "skill": step.skill, "result": result})
            emitter.emit(
                run_id=run_id,
                kind="skill.completed",
                producer="composition_tool",
                payload={"index": idx, "skill": step.skill},
            )
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
                fut = pool.submit(invoke_skill, step.skill, dict(step.inputs))
                fut_map[fut] = (idx, step.skill)

            for fut in as_completed(fut_map):
                idx, skill = fut_map[fut]
                result = fut.result()
                children.append({"index": idx, "skill": skill, "result": result})
                emitter.emit(
                    run_id=run_id,
                    kind="skill.completed",
                    producer="composition_tool",
                    payload={"index": idx, "skill": skill},
                )
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

    return {
        "status": "error",
        "op": parsed.op,
        "children": [],
        "outputs": {},
        "errors": [{"code": "SCP_NOT_IMPLEMENTED", "message": f"op '{parsed.op}' not implemented yet"}],
        "artifacts": [e.model_dump() for e in emitter.list_events()],
    }
