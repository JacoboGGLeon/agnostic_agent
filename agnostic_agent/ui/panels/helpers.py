import streamlit as st
import html
import markdown
import json
import re
from typing import Dict, Any, List, Optional
from agnostic_agent.agent import Agent

def sanitize_display_text(text: Any) -> str:
    """
    Removes frontend artifacts like [object Object] from rendered text.
    Keeps UI clean even if upstream providers/serializers leak JS-like strings.
    """
    if text is None:
        return ""
    out = text if isinstance(text, str) else str(text)
    for _ in range(3):
        decoded = html.unescape(out)
        if decoded == out:
            break
        out = decoded

    # Aggressive cleanup for leaked JS artifacts.
    for _ in range(3):
        out = re.sub(r"(?i),?\s*['\"]?\[object\s*Object\]['\"]?\s*,?", "", out)
        out = re.sub(r"(?i)\[object\s*Object\]", "", out)
        out = out.replace("[object Object]", "")
        out = re.sub(r"(?im)^[\s,'\"-]*\[object\s*Object\][\s,'\"]*$", "", out)
        out = re.sub(r"(?im)^\s*,\s*$", "", out)
    
    out = re.sub(r"(?im)^\s*step\s*\?:\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    # Final strip punctuation leftover from list removals
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"\[\s*,", "[", out)
    out = re.sub(r",\s*\]", "]", out)
    return out.strip()

def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)

def _count_subqueries(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"(?im)^Subquery\s+\d+\s*:", text))

def _build_planner_from_raw_state(raw_state: Dict[str, Any]) -> str:
    planner_trajs = raw_state.get("planner_trajs") or []
    if not isinstance(planner_trajs, list) or not planner_trajs:
        return ""

    lines: List[str] = ["Rol: PLANNER restringe tools+knowledge."]
    scope = raw_state.get("_planner_scope_internal") or {}
    if isinstance(scope, dict) and scope:
        def _fmt_list(items):
            if not isinstance(items, list): return str(items)
            cleaned = []
            for it in items:
                s_it = sanitize_display_text(str(it))
                if s_it: cleaned.append(s_it)
            return "[" + ", ".join(cleaned) + "]"

        skills = _fmt_list(scope.get("active_skills", []))
        tools = _fmt_list(scope.get("allowed_tools", []))
        knowledge = _fmt_list(scope.get("allowed_knowledge", []))
        lines.append(f"Scope: skills={skills}, tools={tools}, knowledge={knowledge}")
        lines.append("")

    for index, traj in enumerate(planner_trajs, start=1):
        if isinstance(traj, dict):
            subquery = _safe_text(traj.get("subquery", ""))
            description = _safe_text(traj.get("description", "")).strip()
        else:
            subquery = _safe_text(getattr(traj, "subquery", ""))
            description = _safe_text(getattr(traj, "description", "")).strip()

        lines.append(f"Subquery {index}: {subquery}")
        lines.append("DAG:")
        if not description:
            lines.append("step 1: (empty)")
        else:
            for raw_line in description.splitlines():
                line = sanitize_display_text(raw_line).strip()
                if not line:
                    continue
                if line.lower().startswith("note:"):
                    lines.append(line)
                elif line.startswith("step "):
                    lines.append(line)
                else:
                    lines.append(f"step ?: {line}")
        if index < len(planner_trajs):
            lines.append("")

    return sanitize_display_text("\n".join(lines))

def normalize_output(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if hasattr(raw, "model_dump"):
        try:
            return raw.model_dump()
        except TypeError:
            pass
    if isinstance(raw, dict):
        return raw
    return {"user_out": str(raw)}

def as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return sanitize_display_text(v)
    if isinstance(v, dict):
        for k in ("final_answer", "text", "content", "answer", "user_out"):
            vv = v.get(k)
            if isinstance(vv, str) and vv.strip():
                return sanitize_display_text(vv)
        return ""
    return sanitize_display_text(v)

def strip_user_prefix(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    prefixes = [
        "Respuesta final (modo usuario):",
        "**Respuesta final (modo usuario):**",
        "RESPUESTA FINAL (modo usuario):",
    ]
    for p in prefixes:
        if t.startswith(p):
            t = t[len(p):].strip()
    return t

def get_raw_state(out: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(out, dict):
        return None
    if isinstance(out.get("messages"), list):
        return out
    rs = out.get("raw_state")
    if isinstance(rs, dict):
        return rs
    dev = out.get("dev_out")
    if isinstance(dev, dict) and isinstance(dev.get("raw_state"), dict):
        return dev["raw_state"]
    return None

def extract_thinking(raw_state: Optional[Dict[str, Any]]) -> str:
    def _coerce_reasoning_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple)):
            parts: List[str] = []
            for item in value:
                txt = _coerce_reasoning_text(item)
                if txt:
                    parts.append(txt)
            return "\n".join(parts).strip()
        if isinstance(value, dict):
            for key in (
                "reasoning_content",
                "reasoning",
                "thoughts",
                "thinking",
                "analysis",
                "text",
                "content",
            ):
                if key in value:
                    txt = _coerce_reasoning_text(value.get(key))
                    if txt:
                        return txt
            try:
                return json.dumps(value, ensure_ascii=False).strip()
            except Exception:
                return str(value).strip()
        if value is None:
            return ""
        return str(value).strip()

    def _extract_think_from_content(content: Any) -> str:
        text = _coerce_reasoning_text(content)
        if not text:
            return ""
        matches = re.findall(r"(?is)<think>(.*?)</think>", text)
        if matches:
            joined = "\n\n".join(m.strip() for m in matches if isinstance(m, str) and m.strip())
            return joined.strip()
        return ""

    def _extract_from_additional_kwargs(ak: Any) -> str:
        if not isinstance(ak, dict):
            return ""
        for key in (
            "reasoning_content",
            "reasoning",
            "thoughts",
            "thinking",
            "analysis",
            "reasoning_text",
        ):
            txt = _coerce_reasoning_text(ak.get(key))
            if txt:
                return txt
        # Provider-specific nested payloads (vLLM/Qwen/OpenAI-like wrappers)
        for nested_key in ("provider_extra", "metadata", "extra", "qwen", "vllm"):
            nested = ak.get(nested_key)
            txt = _coerce_reasoning_text(nested)
            if txt:
                return txt
        return ""

    if not isinstance(raw_state, dict):
        return ""
    msgs = raw_state.get("messages")
    if not isinstance(msgs, list):
        return ""
    
    for m in reversed(msgs):
        if not isinstance(m, dict): continue
        if m.get("type") != "ai": continue
        ak = m.get("additional_kwargs") or {}
        if isinstance(ak, dict) and ak.get("final_answer_thinking"):
            thinking = _extract_from_additional_kwargs(ak)
            if thinking:
                return thinking
    
    for m in reversed(msgs):
        if not isinstance(m, dict): continue
        if m.get("type") != "ai": continue
        ak = m.get("additional_kwargs") or {}
        if isinstance(ak, dict) and ak.get("pipeline_internal"): 
            continue 
        thinking = _extract_from_additional_kwargs(ak)
        if thinking:
            return thinking
        meta = m.get("response_metadata")
        thinking = _coerce_reasoning_text(meta)
        if thinking:
            return thinking
        think_tag = _extract_think_from_content(m.get("content"))
        if think_tag:
            return think_tag
    
    return ""

def extract_summary_deep(raw_state: Optional[Dict[str, Any]], deep_out_text: str) -> str:
    if isinstance(raw_state, dict):
        pipeline_v2 = raw_state.get("pipeline_v2")
        if isinstance(pipeline_v2, dict):
            deep_v2 = pipeline_v2.get("deep_out") or {}
            summary_v2 = deep_v2.get("summary")
            if not isinstance(summary_v2, dict):
                artifacts = deep_v2.get("artifacts") or {}
                if isinstance(artifacts, dict):
                    summary_v2 = artifacts.get("summary_v2")

            if isinstance(summary_v2, dict) and summary_v2:
                lines: List[str] = ["## RESUMEN DEEP DEL PIPELINE (v2)", ""]
                section_order = [
                    ("Analyzer", "analyzer"),
                    ("Planner", "planner"),
                    ("Executor", "executor"),
                    ("Catcher", "catcher"),
                    ("Summarizer", "summarizer"),
                    ("Validator", "validator"),
                    ("Metrics", "metrics"),
                    ("Tool Outputs", "tool_outputs"),
                ]
                for title, key in section_order:
                    section = summary_v2.get(key) or {}
                    if not isinstance(section, dict) or not section:
                        continue
                    lines.append(f"### {title}")
                    for field, value in section.items():
                        rendered = (
                            json.dumps(value, ensure_ascii=False)
                            if isinstance(value, (dict, list))
                            else str(value)
                        )
                        lines.append(
                            f"- {sanitize_display_text(field)}: {sanitize_display_text(rendered)}"
                        )
                    lines.append("")
                return sanitize_display_text("\n".join(lines))

    planner_from_state = ""
    expected_subqueries = 0
    summary: Dict[str, Any] = {}
    if isinstance(raw_state, dict):
        planner_from_state = _build_planner_from_raw_state(raw_state)
        planner_trajs = raw_state.get("planner_trajs") or []
        expected_subqueries = len(planner_trajs) if isinstance(planner_trajs, list) else 0
        summary = raw_state.get("summary") or raw_state.get("pipeline_summary") or {}

    if deep_out_text:
        cleaned = sanitize_display_text(deep_out_text)
        if expected_subqueries > 0 and _count_subqueries(cleaned) < expected_subqueries and planner_from_state:
            deep_out_text = ""
        else:
            return cleaned

    if not summary or not isinstance(summary, dict):
        return ""

    parts = []
    for k in ["analyzer", "planner", "executor", "catcher", "summarizer", "final_answer"]:
        v = summary.get(k, "")
        if k == "planner" and planner_from_state:
            v = planner_from_state
        if isinstance(v, str) and v.strip():
            parts.append(f"**{k.upper()}**\n\n{v.strip()}")
    return sanitize_display_text("\n\n---\n\n".join(parts) if parts else "")

def extract_tool_runs(out: Dict[str, Any], raw_state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(raw_state, dict):
        tr = raw_state.get("tool_runs")
        if isinstance(tr, list):
            return tr
    tr2 = out.get("tool_runs")
    if isinstance(tr2, list):
        return tr2
    dev = out.get("dev_out")
    if isinstance(dev, dict) and isinstance(dev.get("tool_runs"), list):
        return dev["tool_runs"]
    return []

def assistant_messages() -> List[Dict[str, Any]]:
    return [m for m in st.session_state.messages if m.get("role") == "assistant"]

def find_message_by_id(msg_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if msg_id is None:
        return None
    for m in st.session_state.messages:
        if m.get("id") == msg_id:
            return m
    return None

def default_selected_id() -> Optional[int]:
    a = assistant_messages()
    return a[-1]["id"] if a else None

def render_markdown(text: str) -> str:
    """Centralized markdown rendering with sanitization."""
    if not text:
        return ""
    clean_text = sanitize_display_text(text)
    try:
        return markdown.markdown(
            clean_text,
            extensions=["extra", "sane_lists", "nl2br"],
            output_format="html5",
        )
    except Exception:
        return html.escape(clean_text).replace("\n", "<br>")

def card_md(title: str, body_md: str, icon: str = "⬛", hint: str = "") -> None:
    body_md = body_md or "_(vacío)_"
    hint_html = f'<span class="hint">{html.escape(hint)}</span>' if hint else ""
    body_html = render_markdown(body_md)

    st.markdown(
        f"""
<div class="card">
  <div class="card-h">
    <div>{icon} {html.escape(title)}</div>
    {hint_html}
  </div>
  <div class="card-b">{body_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )

def card_code(title: str, code_text: str, icon: str = "🧠", hint: str = "reasoning_content") -> None:
    # Some providers/loggers already HTML-escape text (e.g. &quot;, &#x27;).
    # If we escape again, the UI shows entities literally. Unescape first, then escape once.
    raw = code_text or ""
    for _ in range(3):
        decoded = html.unescape(raw)
        if decoded == raw:
            break
        raw = decoded
    safe = html.escape(raw)
    hint_html = f'<span class="hint">{html.escape(hint)}</span>' if hint else ""
    content = safe if safe.strip() else html.escape("_(no viene thinking en este turno)_")
    st.markdown(
        f"""
<div class="card">
  <div class="card-h">
    <div>{icon} {html.escape(title)}</div>
    {hint_html}
  </div>
  <div class="card-b">
    <div class="codebox">{content}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

def render_tool_runs(tool_runs: List[Dict[str, Any]]) -> None:
    if not tool_runs:
        st.markdown("_(No se ejecutaron tools en este turno.)_")
        return
    st.markdown("#### 🛠 Tools ejecutadas")
    for i, tr in enumerate(tool_runs, start=1):
        if not isinstance(tr, dict):
            st.markdown(f"**{i}. tool_{i}**")
            st.code(str(tr))
            continue
        name = tr.get("name", f"tool_{i}")
        args = tr.get("args", {})
        output = tr.get("output", "")
        st.markdown(f"**{i}. {name}**")
        if args:
            st.code(args, language="json")
        if output != "":
            st.markdown("**Salida:**")
            st.code(str(output))

def next_id() -> int:
    st.session_state.msg_counter += 1
    return st.session_state.msg_counter
