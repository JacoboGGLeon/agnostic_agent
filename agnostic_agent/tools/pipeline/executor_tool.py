from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from langchain_core.messages import ToolMessage


def _resolve_dependency_arg(val: Any, results: Dict[str, Any]) -> Any:
    if isinstance(val, str) and val.strip().startswith("$"):
        ref = val.strip()[1:]
        parts = ref.split(".")
        step_id = parts[0]
        if step_id in results:
            res = results[step_id]
            if len(parts) > 1:
                field = parts[1]
                if field == "output":
                    return res
                if isinstance(res, dict):
                    return res.get(field, val)
            return res
        return val

    if isinstance(val, list):
        return [_resolve_dependency_arg(v, results) for v in val]
    if isinstance(val, dict):
        return {k: _resolve_dependency_arg(v, results) for k, v in val.items()}
    return val


def _repair_tool_args(tool_obj: Any, raw_args: Any) -> Any:
    if not isinstance(raw_args, dict):
        return raw_args

    args = dict(raw_args)
    schema = getattr(tool_obj, "args", None)
    if not isinstance(schema, dict):
        return args

    expected_fields = list(schema.keys())
    if not expected_fields:
        return args

    if "arg_name" in args and "arg_name" not in expected_fields and len(expected_fields) == 1:
        target = expected_fields[0]
        args[target] = args.get("arg_name")
        args.pop("arg_name", None)
        return args

    if len(expected_fields) == 1:
        target = expected_fields[0]
        if target not in args and len(args) == 1:
            only_key = next(iter(args.keys()))
            if only_key not in expected_fields:
                args[target] = args.pop(only_key)

    return args


def execute_executor_tool(
    state: Dict[str, Any],
    *,
    tools: List[Any],
    ai_message_type: Any,
    tool_message_type: Any,
    extract_tool_calls: Callable[[Any], List[Dict[str, Any]]],
    canonical_tool_name: Callable[[Any], str],
    to_jsonable: Callable[[Any], Any],
    json_default: Callable[[Any], Any],
) -> Dict[str, Any]:
    messages = state["messages"]
    dags_by_subquery = state.get("dags_by_subquery") or []
    dag_index: Dict[str, Dict[str, Any]] = {}
    for subq in dags_by_subquery:
        for node in subq.get("dag", []) if isinstance(subq, dict) else []:
            if isinstance(node, dict) and node.get("name"):
                dag_index[str(node.get("name"))] = node
    ai_msgs = [m for m in messages if isinstance(m, ai_message_type)]
    if not ai_msgs:
        return {"messages": [], "executor_steps": [], "artifacts": []}

    ai_plan = ai_msgs[-1]
    tool_calls = getattr(ai_plan, "tool_calls", None)
    if not tool_calls:
        tool_calls = extract_tool_calls(ai_plan)
    if not tool_calls:
        return {"messages": [], "executor_steps": [], "artifacts": []}

    tool_msgs: List[ToolMessage] = []
    exec_steps: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    local_results: Dict[str, Any] = {}

    for tc in tool_calls:
        if isinstance(tc, dict):
            name_raw = tc.get("name")
            args_raw = tc.get("args", {}) or {}
            t_id = tc.get("id")
        else:
            name_raw = getattr(tc, "name", "")
            args_raw = getattr(tc, "args", {}) or {}
            t_id = getattr(tc, "id", "")
        name = canonical_tool_name(name_raw)
        args = _resolve_dependency_arg(args_raw, local_results)

        try:
            tool_obj = next(t for t in tools if t.name == name)
            args = _repair_tool_args(tool_obj, args)
            observation = to_jsonable(tool_obj.invoke(args))
        except StopIteration:
            observation = {"error": f"Tool '{name}' no encontrada."}
        except Exception as exc:
            observation = {"error": f"ExcepciA3n ejecutando tool '{name}': {exc!r}"}

        if t_id:
            local_results[t_id] = observation

        dag_meta = dag_index.get(name, {})
        artifacts.append(
            {
                "artifact_id": f"artifact_{t_id or name}",
                "kind": dag_meta.get("expected_artifact", "tool_output"),
                "producer": name,
                "subquery_id": dag_meta.get("subquery_id", ""),
                "node_id": dag_meta.get("node_id", ""),
                "payload": observation,
            }
        )

        try:
            payload = json.dumps({"value": observation}, ensure_ascii=False, default=json_default)
        except TypeError:
            payload = json.dumps({"value": str(observation)}, ensure_ascii=False)

        tool_msgs.append(tool_message_type(content=payload, tool_call_id=t_id, name=name))
        exec_steps.append(
            {
                "tool_name": name,
                "args": args,
                "tool_call_id": t_id,
                "node_id": dag_meta.get("node_id", ""),
                "subquery_id": dag_meta.get("subquery_id", ""),
                "kind": dag_meta.get("kind", "tool"),
                "expected_artifact": dag_meta.get("expected_artifact", "tool_output"),
            }
        )

    return {"messages": tool_msgs, "executor_steps": exec_steps, "artifacts": artifacts}

