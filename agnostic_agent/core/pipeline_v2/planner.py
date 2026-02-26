from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class PlannedCall:
    name: str
    args: Dict[str, Any]
    subquery_idx: int


def extract_top_level_json_objects(text: str) -> List[str]:
    content = text or ""
    out: List[str] = []
    depth = 0
    start = -1
    for idx, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(content[start : idx + 1])
                    start = -1
    return out


def build_subqueries_from_prompt(prompt: str) -> List[str]:
    clean = (prompt or "").strip()
    if not clean:
        return []
    objects = extract_top_level_json_objects(clean)
    if len(objects) <= 1:
        return [clean]
    prefix = clean.split("{", 1)[0].strip()
    if prefix:
        return [f"{prefix} {obj}".strip() for obj in objects]
    return objects


def _json_key(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)


def dedupe_planned_calls(calls: List[PlannedCall]) -> List[PlannedCall]:
    seen = set()
    out: List[PlannedCall] = []
    for call in calls:
        key = (call.name, _json_key(call.args), call.subquery_idx)
        if key in seen:
            continue
        seen.add(key)
        out.append(call)
    return out


def flatten_tool_calls_by_subquery(raw: List[Dict[str, Any]]) -> List[PlannedCall]:
    """
    Normalize planner output into deterministic call rows.
    Expected input shape:
      [{"subquery_idx": 1, "tool_calls": [{"name":"...", "args": {...}}]}]
    """
    rows: List[PlannedCall] = []
    for block in raw or []:
        subquery_idx = int(block.get("subquery_idx", 0) or 0)
        for call in block.get("tool_calls", []) or []:
            name = str(call.get("name", "")).strip()
            if not name:
                continue
            args = call.get("args", {})
            if not isinstance(args, dict):
                args = {}
            rows.append(
                PlannedCall(
                    name=name,
                    args=args,
                    subquery_idx=subquery_idx,
                )
            )
    return dedupe_planned_calls(rows)
