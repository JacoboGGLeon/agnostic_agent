from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage


def _coerce_content_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or item.get("content", "") or "")
            else:
                parts.append(str(item))
        return "".join(parts)
    return "" if content is None else str(content)


def _scan_balanced_json(s: str, i: int) -> Tuple[Optional[str], int]:
    if i < 0 or i >= len(s) or s[i] != "{":
        return None, i

    depth = 0
    in_str = False
    esc = False
    start = i

    while i < len(s):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1], i + 1
        i += 1

    return None, i


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

    for call in raw_calls:
        if isinstance(call, dict):
            fn = call.get("function") or {}
            name = call.get("name") or fn.get("name") or call.get("tool_name")
            if "args" in call:
                args_raw = call.get("args")
            else:
                args_raw = fn.get("arguments") or call.get("arguments") or call.get("parameters")
            id_ = call.get("id") or call.get("tool_call_id")
        else:
            fn = getattr(call, "function", None)
            name = (
                getattr(call, "name", None)
                or (getattr(fn, "name", None) if fn else None)
                or getattr(call, "tool_name", None)
            )
            args_raw = (
                getattr(call, "args", None)
                or (getattr(fn, "arguments", None) if fn else None)
                or getattr(call, "arguments", None)
                or getattr(call, "parameters", None)
            )
            id_ = getattr(call, "id", None) or getattr(call, "tool_call_id", None)

        args = _parse_args_maybe_json(args_raw)
        norm_name = _canonical_tool_name(name)
        if norm_name:
            norm.append({"id": id_ or f"call_{uuid.uuid4().hex}", "name": norm_name, "args": args})
    return norm


def _extract_tool_calls_via_etree(text: str) -> List[Dict[str, Any]]:
    wrapped = f"<root>{text}</root>"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return []

    out: List[Dict[str, Any]] = []
    for node in root.findall(".//tool_call"):
        raw = "".join(node.itertext()).strip()
        if not raw:
            continue

        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out.extend([it for it in obj if isinstance(it, dict)])
            continue
        except Exception:
            pass

        j = raw.find("{")
        if j != -1:
            js, _ = _scan_balanced_json(raw, j)
            if js:
                try:
                    obj2 = json.loads(js)
                    if isinstance(obj2, dict):
                        out.append(obj2)
                except Exception:
                    pass

    return out


def _extract_tool_calls_via_xmlish_bracescan(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tag_open = "<tool_call>"
    tag_close = "</tool_call>"

    pos = 0
    while True:
        a = text.find(tag_open, pos)
        if a == -1:
            break
        b = text.find(tag_close, a)
        if b == -1:
            break

        chunk = text[a + len(tag_open) : b].strip()
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                out.append(obj)
            elif isinstance(obj, list):
                out.extend([it for it in obj if isinstance(it, dict)])
        except Exception:
            j = chunk.find("{")
            if j != -1:
                js, _ = _scan_balanced_json(chunk, j)
                if js:
                    try:
                        obj2 = json.loads(js)
                        if isinstance(obj2, dict):
                            out.append(obj2)
                    except Exception:
                        pass

        pos = b + len(tag_close)

    return out


def _extract_xml_tool_calls(ai_msg: AIMessage) -> List[Dict[str, Any]]:
    text = _coerce_content_str(getattr(ai_msg, "content", ""))
    if "<tool_call" not in text:
        return []

    parsed = _extract_tool_calls_via_etree(text)
    if not parsed:
        parsed = _extract_tool_calls_via_xmlish_bracescan(text)

    calls: List[Dict[str, Any]] = []
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool_name")
        args_raw = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
        args = _parse_args_maybe_json(args_raw)
        if name:
            calls.append({"id": f"call_{uuid.uuid4().hex}", "name": name, "args": args})
    return calls


def extract_tool_calls_from_jsonish_text(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    calls: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()

    def _append(name_raw: Any, args_raw: Any) -> None:
        name = _canonical_tool_name(name_raw)
        args = _parse_args_maybe_json(args_raw)
        if not name:
            return
        try:
            args_key = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except Exception:
            args_key = repr(args)
        key = (name, args_key)
        if key in seen:
            return
        seen.add(key)
        calls.append({"id": f"call_{uuid.uuid4().hex}", "name": name, "args": args})

    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        js, next_i = _scan_balanced_json(text, i)
        if not js:
            i += 1
            continue
        i = max(next_i, i + 1)
        try:
            obj = json.loads(js)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        tool_uses = obj.get("tool_uses")
        if isinstance(tool_uses, list):
            for item in tool_uses:
                if not isinstance(item, dict):
                    continue
                _append(
                    item.get("recipient_name") or item.get("name") or item.get("tool_name"),
                    item.get("parameters") or item.get("args") or item.get("arguments") or {},
                )
        elif "tool_uses" not in obj:
            _append(
                obj.get("recipient_name") or obj.get("name") or obj.get("tool_name"),
                obj.get("parameters") or obj.get("args") or obj.get("arguments") or {},
            )

    return calls


def extract_tool_calls(ai_msg: AIMessage) -> List[Dict[str, Any]]:
    if not isinstance(ai_msg, AIMessage):
        return []

    tc = getattr(ai_msg, "tool_calls", None)
    norm = _normalize_toolcalls_list(tc)
    if norm:
        return norm

    addkw = getattr(ai_msg, "additional_kwargs", {}) or {}
    tc2 = addkw.get("tool_calls")
    norm2 = _normalize_toolcalls_list(tc2)
    if norm2:
        return norm2

    xml_calls = _extract_xml_tool_calls(ai_msg)
    if xml_calls:
        return xml_calls

    content_text = _coerce_content_str(getattr(ai_msg, "content", ""))
    return extract_tool_calls_from_jsonish_text(content_text)
