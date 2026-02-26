from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, AnyMessage

from agnostic_agent.graph.summarization import (
    build_agnostic_user_answer,
    looks_like_technical_answer,
    summarize_tool_runs as summarize_tool_runs_shared,
    summarize_tool_runs_compact as summarize_tool_runs_compact_shared,
)
from agnostic_agent.graph.tool_call_parsing import (
    extract_tool_calls as extract_tool_calls_shared,
    extract_tool_calls_from_jsonish_text as extract_tool_calls_from_jsonish_text_shared,
)


def _coerce_content_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", "") or p.get("content", "") or "")
            else:
                parts.append(str(p))
        return "".join(parts)
    return "" if content is None else str(content)


def _parse_args_maybe_json(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _canonical_tool_name(name: Any) -> str:
    if name is None:
        return ""
    raw = str(name).strip()
    if not raw:
        return ""
    lowered = raw.lower()
    for prefix in ("functions.", "function.", "tools.", "tool."):
        if lowered.startswith(prefix):
            return raw[len(prefix) :].strip()
    if "." in raw:
        return raw.split(".")[-1].strip()
    return raw


def _normalize_toolcalls_list(raw_calls: Any) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    if not raw_calls:
        return norm
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    elif not isinstance(raw_calls, list):
        raw_calls = [raw_calls]

    for c in raw_calls:
        if isinstance(c, dict):
            fn = c.get("function") or {}
            name = c.get("name") or fn.get("name") or c.get("tool_name")
            if "args" in c:
                args_raw = c.get("args")
            else:
                args_raw = fn.get("arguments") or c.get("arguments") or c.get("parameters")
            id_ = c.get("id") or c.get("tool_call_id")
        else:
            fn = getattr(c, "function", None)
            name = (
                getattr(c, "name", None)
                or (getattr(fn, "name", None) if fn else None)
                or getattr(c, "tool_name", None)
            )
            args_raw = (
                getattr(c, "args", None)
                or (getattr(fn, "arguments", None) if fn else None)
                or getattr(c, "arguments", None)
                or getattr(c, "parameters", None)
            )
            id_ = getattr(c, "id", None) or getattr(c, "tool_call_id", None)

        args = _parse_args_maybe_json(args_raw)
        norm_name = _canonical_tool_name(name)
        if norm_name:
            norm.append({"id": id_ or f"call_{uuid.uuid4().hex}", "name": norm_name, "args": args})
    return norm


def _extract_tool_calls_from_jsonish_text(text: str) -> List[Dict[str, Any]]:
    return extract_tool_calls_from_jsonish_text_shared(text)


def extract_tool_calls(ai_msg: AIMessage) -> List[Dict[str, Any]]:
    return extract_tool_calls_shared(ai_msg)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return _to_jsonable(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _to_jsonable(vars(obj))
        except Exception:
            pass
    return str(obj)


def _decode_tool_content(raw: Any) -> Any:
    if isinstance(raw, list):
        if len(raw) == 1:
            raw = raw[0]
        else:
            return _to_jsonable(raw)
    if isinstance(raw, dict):
        if "value" in raw and len(raw) == 1:
            return _to_jsonable(raw["value"])
        return _to_jsonable(raw)
    if not isinstance(raw, str):
        return _to_jsonable(raw)
    s = raw.strip()
    if not s:
        return ""
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and "value" in parsed and len(parsed) == 1:
            return _to_jsonable(parsed["value"])
        return _to_jsonable(parsed)
    except Exception:
        return s


_THINK_RE = re.compile(r"(?is)<think>.*?(?:</think>|$)\s*")


def strip_think(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    cleaned = _THINK_RE.sub("", txt).strip()
    if not cleaned and txt.strip():
        return ""
    return cleaned


def _is_pipeline_internal_ai(m: AnyMessage) -> bool:
    if not isinstance(m, AIMessage):
        return False
    addkw = getattr(m, "additional_kwargs", {}) or {}
    if addkw.get("pipeline_internal") is True:
        return True
    txt = _coerce_content_str(getattr(m, "content", "")).lstrip()
    if txt.startswith("## Resumen del pipeline"):
        return True
    if txt.startswith("## Resumen deep del pipeline"):
        return True
    if txt.startswith("### VALIDATOR"):
        return True
    return False


def find_last_assistant_real(messages: List[AnyMessage]) -> Optional[AIMessage]:
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) and not _is_pipeline_internal_ai(m):
            txt = _coerce_content_str(getattr(m, "content", "")).strip()
            if txt:
                return m
    return None


def summarize_tool_runs(user_text: str, runs: List[Dict[str, Any]]) -> str:
    return summarize_tool_runs_shared(user_text, runs, json_default=_json_default)


def summarize_tool_runs_compact(runs: List[Dict[str, Any]]) -> str:
    return summarize_tool_runs_compact_shared(runs)


def build_user_answer_from_runs(user_prompt: str, runs: List[Dict[str, Any]]) -> str:
    return build_agnostic_user_answer(user_prompt, runs)


def is_technical_answer(text: str) -> bool:
    return looks_like_technical_answer(text)


def _sanitize_subquery_text(text: Any) -> str:
    t = _coerce_content_str(text).strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\brealiza\s+la\s+concili\S*\s+del\s+cr\S*\s*$", "", t, flags=re.IGNORECASE).strip()
    return t.strip(" .")


def _is_placeholder_subquery(text: Any) -> bool:
    t = _coerce_content_str(text).strip().lower()
    if not t:
        return True
    placeholder_patterns = [
        r"^q\d+$",
        r"^subquery\s*\d+$",
        r"^paso/?pregunta\s*\d+$",
        r"^paso\s*\d+$",
        r"^pregunta\s*\d+$",
    ]
    return any(re.match(p, t) for p in placeholder_patterns)


def _extract_top_level_json_objects(text: Any) -> List[str]:
    source = _coerce_content_str(text)
    if not source:
        return []
    items: List[str] = []
    depth = 0
    in_str = False
    escape = False
    start = -1
    for idx, ch in enumerate(source):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = source[start : idx + 1].strip()
                    if chunk:
                        items.append(chunk)
                    start = -1
    return items


def _resolve_effective_skills(
    state: Dict[str, Any],
    skill_registry: Any | None = None,
) -> List[str]:
    active = state.get("_active_skills_internal") or []
    if isinstance(active, list) and active:
        return [str(s) for s in active if str(s).strip()]
    forced_skill = state.get("forced_skill")
    allow = state.get("skills_allowlist") or []
    if forced_skill and forced_skill != "Auto (Analyzer)":
        allow = [forced_skill]
    resolved: List[str] = []
    if isinstance(allow, list):
        for s in allow:
            s_name = str(s).strip()
            if not s_name or s_name == "Auto (Analyzer)":
                continue
            if skill_registry is None or skill_registry.get_skill(s_name):
                resolved.append(s_name)
    return resolved


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _is_ai_with_tool_calls(m: AnyMessage) -> bool:
    if not isinstance(m, AIMessage):
        return False
    return bool(extract_tool_calls(m))
