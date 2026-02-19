import streamlit as st
import html
import markdown
import json
from typing import Dict, Any, List, Optional
from agnostic_agent.agent import Agent

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
        return v.strip()
    if isinstance(v, dict):
        for k in ("final_answer", "text", "content", "answer", "user_out"):
            vv = v.get(k)
            if isinstance(vv, str) and vv.strip():
                return vv.strip()
        return ""
    return str(v).strip()

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
    if not isinstance(raw_state, dict):
        return ""
    msgs = raw_state.get("messages")
    if not isinstance(msgs, list):
        return ""
    
    for m in reversed(msgs):
        if not isinstance(m, dict): continue
        if m.get("type") != "ai": continue
        ak = m.get("additional_kwargs") or {}
        if isinstance(ak, dict):
            if ak.get("final_answer_thinking"):
                thinking = ak.get("reasoning_content") or ak.get("reasoning") or ak.get("thoughts") or ""
                return thinking.strip() if isinstance(thinking, str) else ""
    
    for m in reversed(msgs):
        if not isinstance(m, dict): continue
        if m.get("type") != "ai": continue
        ak = m.get("additional_kwargs") or {}
        if isinstance(ak, dict) and ak.get("pipeline_internal"): continue 
        thinking = ak.get("reasoning_content") or ak.get("reasoning") or ak.get("thoughts") or ""
        if thinking and isinstance(thinking, str) and thinking.strip():
            return thinking.strip()
    
    return ""

def extract_summary_deep(raw_state: Optional[Dict[str, Any]], deep_out_text: str) -> str:
    if deep_out_text:
        return deep_out_text
    if not isinstance(raw_state, dict):
        return ""
    summary = raw_state.get("summary") or raw_state.get("pipeline_summary")
    if not isinstance(summary, dict):
        return ""
    parts = []
    for k in ["analyzer", "planner", "executor", "catcher", "summarizer", "final_answer"]:
        v = summary.get(k, "")
        if isinstance(v, str) and v.strip():
            parts.append(f"**{k.upper()}**\n\n{v.strip()}")
    return "\n\n---\n\n".join(parts) if parts else ""

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

def card_md(title: str, body_md: str, icon: str = "⬛", hint: str = "") -> None:
    body_md = body_md or "_(vacío)_"
    hint_html = f'<span class="hint">{html.escape(hint)}</span>' if hint else ""
    try:
        body_html = markdown.markdown(body_md, extensions=['extra'])
    except Exception:
        body_html = html.escape(body_md).replace("\n", "<br>")

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
    safe = html.escape(code_text or "")
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
