from __future__ import annotations

import os
import json
import html
from typing import Any, Dict, Optional, List, Tuple

import streamlit as st
from agnostic_agent.agent import Agent
from agnostic_agent.capabilities import PlannerConfig
from agnostic_agent.tools import get_default_tools

# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="Agnostic Agent · Chat Studio (Inspector)",
    page_icon="🧪",
    layout="wide",
)

# -----------------------------
# Hide Streamlit chrome (dark cintillo)
# -----------------------------
st.markdown("""
<style>
[data-testid="stHeader"] { 
    /* display: none !important; */
    background: transparent;
    color: var(--text);
}
/* [data-testid="stToolbar"] { display: none !important; } */
[data-testid="stDecoration"] { display: none !important; }
/* #MainMenu { visibility: hidden !important; } */
/* footer { visibility: hidden !important; } */
.block-container { padding-top: 3rem !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CSS (Studio + Inspector layout)
# -----------------------------
st.markdown(
    """
<style>
:root{
  --bg: #0b1020;
  --panel: rgba(255,255,255,.06);
  --panel2: rgba(255,255,255,.08);
  --border: rgba(255,255,255,.10);
  --text: rgba(255,255,255,.92);
  --muted: rgba(255,255,255,.65);
  --accent: #7c5cff;
  --good: #2dd4bf;

  --r: 18px;
  --r2: 14px;
  --shadow: 0 12px 35px rgba(0,0,0,.35);
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

.stApp{
  background:
    radial-gradient(1200px 500px at 10% -10%, rgba(124,92,255,.35), transparent 60%),
    radial-gradient(900px 500px at 90% 0%, rgba(45,212,191,.18), transparent 60%),
    linear-gradient(180deg, var(--bg), #070a14 60%, #050712);
  color: var(--text);
}

.block-container{ padding-top: 1.0rem; padding-bottom: 1.6rem; }

section[data-testid="stSidebar"]{
  background: rgba(0,0,0,.18);
  border-right: 1px solid rgba(255,255,255,.06);
}

.topbar{
  display:flex; align-items:center; justify-content:space-between;
  gap:12px;
  padding: 12px 14px;
  border-radius: var(--r);
  background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.05));
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  margin-bottom: 10px;
}
.brand{display:flex; align-items:center; gap:10px;}
.logo{
  width: 38px; height: 38px; border-radius: 12px;
  display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, rgba(124,92,255,.9), rgba(45,212,191,.6));
  box-shadow: 0 10px 25px rgba(124,92,255,.22);
  font-size: 18px;
}
.title{font-size: 15px; font-weight: 800; line-height: 1.1;}
.subtitle{font-size: 12px; color: var(--muted);}

.badges{display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end;}
.badge{
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,.06);
  color: var(--text);
}
.badge.accent{ border-color: rgba(124,92,255,.45); }
.badge.good{ border-color: rgba(45,212,191,.45); }

.card{
  border-radius: var(--r);
  border: 1px solid var(--border);
  background: rgba(255,255,255,.06);
  box-shadow: 0 10px 30px rgba(0,0,0,.28);
  overflow: hidden;
  margin-top: 6px;
}
.card .card-h{
  display:flex; align-items:center; justify-content:space-between;
  gap:10px;
  padding: 10px 12px;
  background: rgba(255,255,255,.05);
  border-bottom: 1px solid var(--border);
  font-weight: 800;
  font-size: 13px;
}
.card .card-h .hint{
  font-weight: 500; font-size: 11px; color: var(--muted);
}
.card .card-b{
  padding: 12px 12px 10px 12px;
  font-size: 14px;
  color: var(--text);
}

/* Code-like block inside cards (for Thinking) */
.codebox{
  margin-top: 8px;
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(0,0,0,.28);
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
  color: rgba(255,255,255,.92);
}

/* Right-align assistant messages (best effort assumption: alternating user/ai) */
div[data-testid="stChatMessage"]:nth-child(even) {
    flex-direction: row-reverse;
    background-color: rgba(255,255,255,0.02);
}
div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stMarkdown"] {
    text-align: right;
}

/* User bubble */
.bubble-user{
  padding: 10px 12px;
  border-radius: 16px;
  border: 1px solid rgba(124,92,255,.35);
  background: linear-gradient(180deg, rgba(124,92,255,.22), rgba(255,255,255,.05));
  box-shadow: 0 8px 24px rgba(0,0,0,.25);
}

/* Inspector wrapper */
.inspector{
  border-radius: var(--r);
  border: 1px solid var(--border);
  background: rgba(255,255,255,.05);
  box-shadow: var(--shadow);
  box-shadow: var(--shadow);
  padding: 12px;
}
.inspector-box{
  border-radius: var(--r);
  border: 1px solid var(--border);
  background: rgba(255,255,255,.05);
  box-shadow: var(--shadow);
  padding: 16px;
  margin-bottom: 20px;
}
.inspector h3{ margin: 0 0 6px 0; }

/* Expanders */
[data-testid="stExpander"]{
  border-radius: var(--r);
  border: 1px solid var(--border);
  background: rgba(255,255,255,.04);
  overflow:hidden;
}

/* Chat spacing */
[data-testid="stChatMessage"]{
  padding-top: 0.25rem;
  padding-bottom: 0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Session state
# -----------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "tools_strict"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "msg_counter" not in st.session_state:
    st.session_state.msg_counter = 0
if "selected_msg_id" not in st.session_state:
    st.session_state.selected_msg_id = None
if "export_json" not in st.session_state:
    st.session_state.export_json = None

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    # Sidebar Header
    st.markdown("### Agnostic Agent · Settings")

    st.markdown("#### 🧭 Inspector")
    show_inspector = st.toggle("Activar Inspector", value=True)
    
    if show_inspector:
        st.caption("Vistas:")
        show_thinking_tab = st.checkbox("🧠 Thinking", value=True)
        show_deep_tab = st.checkbox("🧠 Deep", value=True)
        show_dev_tab = st.checkbox("🔍 Dev", value=True)
    
    st.divider()

    # Session / Transcript
    st.caption(f"Mensajes: {len(st.session_state.messages)}")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent = None
            st.session_state.selected_msg_id = None
            st.toast("Chat reiniciado.", icon="🧹")
            st.rerun()
            
    with c2:
        if st.button("⬇️ Export", use_container_width=True):
            export = {"messages": st.session_state.messages}
            st.session_state.export_json = json.dumps(export, ensure_ascii=False, indent=2)
            st.toast("Transcript listo.", icon="⬇️")

    if isinstance(st.session_state.export_json, str):
        st.download_button(
            "Descargar JSON",
            data=st.session_state.export_json,
            file_name="transcript.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    # Models Section
    st.markdown("#### 🤖 Models")
    
    llm_name = os.getenv("LLM_SERVED_NAME", "qwen2.5-14b-instruct")
    emb_name = os.getenv("EMB_SERVED_NAME", "Qwen/Qwen3-Embedding-0.6B")
    
    st.caption("Planner / Main LLM")
    planner_model_name = st.text_input("Model Name", value=llm_name, disabled=True)
    
    st.caption("Embedding Server")
    st.text_input("Emb Name", value=emb_name, disabled=True)

    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    # Removed old session info block (merged above)

# (Mode change logic removed)

# -----------------------------
# Helpers
# -----------------------------
def next_id() -> int:
    st.session_state.msg_counter += 1
    return st.session_state.msg_counter

def get_or_init_agent() -> Agent:
    if st.session_state.agent is None:
        with st.spinner(f"Inicializando agente (Unified Mode)…"):
            try:
                # CREAMOS CONFIG DE PLANNER EXPLÍCITA (Unified Mode default)
                # Usamos las variables globales definidas en el sidebar (top-level scope)
                cfg = PlannerConfig(
                    model_name=planner_model_name if 'planner_model_name' in globals() else None,
                    temperature=temperature if 'temperature' in globals() else 0.0,
                    max_steps=16,
                )
                
                # Inicializamos pasando esa config
                st.session_state.agent = Agent.init(config_or_setup=cfg)

                # Sync Skills Config (persistence across resets)
                if "skills_config" in st.session_state and st.session_state.agent.skill_registry:
                    for sname, senabled in st.session_state.skills_config.items():
                        st.session_state.agent.skill_registry.set_enabled(sname, senabled)

            except Exception as e:
                st.error(f"Error iniciando agente: {e}")
                st.stop()
    return st.session_state.agent

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
        if not isinstance(m, dict):
            continue
        if m.get("type") != "ai":
            continue
        ak = m.get("additional_kwargs") or {}
        if isinstance(ak, dict) and ak.get("pipeline_internal"):
            continue
        thinking = ak.get("reasoning_content") or ak.get("reasoning") or ak.get("thoughts") or ""
        return thinking.strip() if isinstance(thinking, str) else ""
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
    # NOTE: body_md here is treated as plain HTML content; for Deep this is OK.
    # For Thinking we use code-card below so it looks like "markdown blocks".
    st.markdown(
        f"""
<div class="card">
  <div class="card-h">
    <div>{icon} {html.escape(title)}</div>
    {hint_html}
  </div>
  <div class="card-b">{body_md}</div>
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

# If nothing selected yet, default to last assistant
if st.session_state.selected_msg_id is None:
    st.session_state.selected_msg_id = default_selected_id()

# -----------------------------
# Top bar
# -----------------------------
st.markdown(
    f"""
<div class="topbar">
  <div class="brand">
    <div class="logo">🧪</div>
    <div>
      <div class="title">Agnostic Agent · Chat Studio</div>
    </div>
  </div>
  <div class="badges">
    <!-- Unified Mode badge removed -->
    <!-- <span class="badge">🔎 inspector: on</span> -->
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Main layout: Tabs for Online / Offline
# -----------------------------
tab_online, tab_offline = st.tabs(["💬 Online Chat", "🛠 Offline Manager"])

# ==========================================
# TAB 1: ONLINE CHAT
# ==========================================
with tab_online:
    feed_col, insp_col = st.columns([2.2, 1.0], gap="large")

    # -------- FEED (left) --------
    with feed_col:
        for msg in st.session_state.messages:
            role = msg.get("role", "user")

            if role == "user":
                with st.chat_message("user"):
                    # Use standard markdown for user messages to support rich text
                    st.markdown(msg.get("content",""))

            elif role == "assistant":
                out = msg.get("out") or {}
                content = msg.get("content") or ""
                # Try to get mode from 'out', fallback to session state if current run, else '?'
                used_mode = out.get("agent_mode", "")
                
                raw_state = get_raw_state(out)
                tool_runs = extract_tool_runs(out, raw_state)

                # --- Capture Tool Logs for Offline Manager ---
                if "tool_logs" not in st.session_state:
                    st.session_state["tool_logs"] = []
                
                import datetime
                for tr in tool_runs:
                    log_entry = {
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "tool": tr.get("name", "unknown"),
                        "input": tr.get("args", {}),
                        "output": tr.get("output", "")
                    }
                    st.session_state["tool_logs"].append(log_entry)
                
                badge_html = f'<span class="badge" style="font-size:10px; padding:2px 6px; margin-left:8px; opacity:0.7;">{used_mode}</span>' if used_mode else ""
                
                with st.chat_message("assistant"):
                    # Pretty answer only
                    card_md(
                        title=f"Respuesta {badge_html}",
                        body_md=html.escape(content or "_(sin respuesta)_").replace("\n", "<br>"),
                        icon="👤",
                        hint=f"id={msg.get('id')}",
                    )

                    c1, c2, c3 = st.columns([1.2, 1.0, 0.8])
                    with c1:
                        st.caption(f"🛠 tools: {len(tool_runs)}")
                    with c2:
                        pass # st.caption("📎 Inspector →")
                    with c3:
                        if st.button("🔎 Inspect", key=f"inspect_{msg.get('id')}", use_container_width=True):
                            st.session_state.selected_msg_id = msg.get("id")
                            st.toast(f"Inspector → id={msg.get('id')}", icon="🔎")
                            st.rerun()

    # -------- INSPECTOR (right) --------
    with insp_col:
        # Wrap everything in a container div
        st.markdown('<div class="inspector-box">', unsafe_allow_html=True)
        st.markdown("### 🔎 Inspector")
        
        # Only show content if enabled in sidebar
        if not show_inspector:
            st.info("Inspector oculto. Actívalo en la barra lateral.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            a_msgs = assistant_messages()
            if not a_msgs:
                st.info("Aún no hay respuestas del agente. Escribe algo para empezar.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                ids = [m["id"] for m in a_msgs]

                def label(mid: int) -> str:
                    m = find_message_by_id(mid) or {}
                    out = m.get("out") or {}
                    text = strip_user_prefix(as_text(out.get("user_out"))).replace("\n", " ").strip()
                    text = (text[:60] + "…") if len(text) > 60 else text
                    return f"id={mid} · {text or '(sin texto)'}"

                if st.session_state.selected_msg_id not in ids:
                    st.session_state.selected_msg_id = ids[-1]

                sel = st.selectbox(
                    "Mensaje seleccionado",
                    options=ids,
                    index=ids.index(st.session_state.selected_msg_id),
                    format_func=label,
                    key="inspector_selectbox",
                )
                st.session_state.selected_msg_id = sel

                m = find_message_by_id(st.session_state.selected_msg_id) or {}
                out = m.get("out") or {}
                raw_state = get_raw_state(out)

                thinking = extract_thinking(raw_state)
                deep_txt = extract_summary_deep(raw_state, as_text(out.get("deep_out")))
                tool_runs = extract_tool_runs(out, raw_state)

                tab_specs: List[Tuple[str, str]] = []
                if show_thinking_tab:
                    tab_specs.append(("🧠 Thinking", "thinking"))
                if show_deep_tab:
                    tab_specs.append(("🧠 Deep", "deep"))
                if show_dev_tab:
                    tab_specs.append(("🔍 Dev", "dev"))

                tabs = st.tabs([t[0] for t in tab_specs])

                for (tab_title, tab_key), tab in zip(tab_specs, tabs):
                    with tab:
                        if tab_key == "thinking":
                            card_code("Pensamiento (thinking)", thinking, icon="🧠", hint="reasoning_content")

                        elif tab_key == "deep":
                            # Render Deep/Summary with real Markdown support
                            # We split the card HTML to inject the markdown in between
                            st.markdown("### 🧠 Vista profunda (deep_out / summary)")
                            st.caption("pipeline")
                            
                            if deep_txt:
                                st.markdown(deep_txt)
                            else:
                                st.markdown("_(vacío)_")

                        elif tab_key == "dev":
                            render_tool_runs(tool_runs)

                            with st.expander("🧬 raw_state (debug)", expanded=False):
                                if isinstance(raw_state, dict) and raw_state:
                                    st.json(raw_state)
                                else:
                                    st.markdown("_(sin raw_state)_")

            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Chat input (bottom of Online tab)
    # -----------------------------
    prompt = st.chat_input("Escribe tu mensaje…")

    if prompt:
        uid = next_id()
        # Prepare message payload
        # Start with empty 'out' but include the current mode for badge display
        msg_payload = {
            "id": uid, 
            "role": "user", 
            "content": prompt, 
            "out": {"agent_mode": "unified"} 
        }
        st.session_state.messages.append(msg_payload)

        agent = get_or_init_agent()
        try:
            # Pass full params if strict mode or updated logic requires
            raw_out = agent.run_turn(prompt)
        except Exception as e:
            st.error(f"Error corriendo agente: {e}")
            st.stop()

        out = normalize_output(raw_out)
        aid = next_id()
        st.session_state.messages.append(
            {"id": aid, "role": "assistant", "content": strip_user_prefix(as_text(out.get("user_out"))), "out": out}
        )

        st.session_state.selected_msg_id = aid
        st.rerun()


# ==========================================
# TAB 2: OFFLINE MANAGER
# ==========================================
with tab_offline:
    # Instantiate agent (the tool config is applied inside the agent init or run loop if logic permits, 
    # but here we update the instance's tool list for hot-swapping if possible)
    
    agent = get_or_init_agent()
    
    # Apply tools_config dynamically:
    if "tools_config" in st.session_state:
        enabled_tools = [name for name, active in st.session_state.tools_config.items() if active]
        # Re-bind logic might be complex if LangGraph compiles the graph with tools fixed.
        # But for 'agent.tools' list it's fine.
        # The key is: get_or_init_agent creates a NEW agent if st.session_state.agent is None.
        # Our toggle callback sets it to None, so this block will just re-init correctly.
        pass 
    
    # Instantiate agent with logging wrapper if we want to capture logs (TODO: Implement callback/wrapper)
    # For now, we will just use the standard instantiation but we need to hook into it for logging.
    # Since we can't easily hook without changing Agent class, we'll rely on the Agent's output 
    # to populate 'tool_logs' in the processing loop (tab_online).
    
    agent = get_or_init_agent()
    
    # If we want to apply the tools_config dynamically:
    if "tools_config" in st.session_state:
        enabled_tools = [name for name, active in st.session_state.tools_config.items() if active]
        agent.tools = get_default_tools(enabled_tools)
        agent.tools_map = {t.name: t for t in agent.tools}
    # Create sub-tabs
    tab_km, tab_tm, tab_skills, tab_logs = st.tabs(["📚 Knowledge Manager", "🛠 Tools Manager", "🧩 Skills Manager", "📜 Logs de Ejecución"])

    # -------------------------------------------------------------------------
    # 📚 Knowledge Manager Tab
    # -------------------------------------------------------------------------
    with tab_km:
        st.markdown("### 📚 Gestor de Conocimiento")
        st.info(
            "Aquí puedes subir documentos PDF para procesarlos (Parsing → Chunking → Embedding → SQLite-Vec). "
            "Esto poblará la base de datos local `embeddings.db`."
        )

        # File uploader
        uploaded_file = st.file_uploader("Subir documento PDF", type=["pdf"])
        
        # DB path configuration (env vars with local defaults)
        DB_PATH = os.getenv("AGNOSTIC_DB_PATH", os.path.join(os.getcwd(), "embeddings.db"))
        DOCS_DIR = os.getenv("AGNOSTIC_DOCS_DIR", os.path.join(os.getcwd(), "documents"))
        os.makedirs(DOCS_DIR, exist_ok=True)
        
        if uploaded_file is not None:
            # Save to temp/docs dir
            # Use absolute path for clarity and robustness
            save_path = os.path.abspath(os.path.join(DOCS_DIR, uploaded_file.name))
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.success(f"Archivo guardado en: `{save_path}`")
            
            if st.button("🚀 Procesar e Ingestar", type="primary"):
                from agnostic_agent.knowledge.vector import ingest_pdf_file
                
                progress_bar = st.progress(0.0, text="Iniciando...")
                
                def _streamlit_progress_cb(p: float, msg: str):
                    progress_bar.progress(p, text=msg)

                # with st.spinner("Procesando documento... (puede tardar si usa CPU)"):
                try:
                    res = ingest_pdf_file(save_path, DB_PATH, progress_callback=_streamlit_progress_cb)
                    if "error" in res:
                        st.error(f"Error: {res['error']}")
                        st.info("Nota: Si 'Docling' o 'PyMuPDF' no están instalados, no se podrá extraer texto.")
                    else:
                        st.balloons()
                        st.success("¡Ingesta completada!")
                        st.json(res)
                        
                        # Log to history
                        from agnostic_agent.knowledge.vector import log_ingestion_event
                        
                        # Construct metadata
                        meta = {
                            "file": uploaded_file.name,
                            "chunks": res.get("chunks_inserted", 0),
                            "db_path": DB_PATH,
                            "status": "success"
                        }
                        # JSONL file in same dir as docs
                        history_file = os.path.join(DOCS_DIR, "knowledge_history.jsonl")
                        log_ingestion_event(meta, history_file)
                        
                except Exception as e:
                    st.error(f"Excepción crítica: {e}")

    st.divider()
    
    # DB Stats
    st.markdown("### 📊 Estado de la Base de Datos")
    try:
        # FORCE RELOAD to ensure new functions are picked up
        import sys
        import importlib
        import agnostic_agent.knowledge.vector
        importlib.reload(agnostic_agent.knowledge.vector)
        from agnostic_agent.knowledge.vector import get_stats, get_ingestion_history
        
        stats = get_stats(DB_PATH)
        
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Chunks / Vectores", f"{stats.get('vector_count', 0)}")
        s2.metric("Archivos", stats.get("files", 0))
        
        # Format bytes to MB
        sz = stats.get("size_bytes", 0)
        sz_mb = f"{sz / (1024*1024):.2f} MB"
        s3.metric("Tamaño en Disco", sz_mb)
        
        s4.metric("Dimensiones", stats.get("dim", 0))
        
        st.info("💡 **Tip:** Para consultar esta base de conocimiento, ¡simplemente pregúntale al agente! Él decidirá cuándo usar la herramienta `search_knowledge_base`.")
        
        st.markdown("#### 📜 Historial de Ingesta (Persistente)")
        history_file = os.path.join(DOCS_DIR, "knowledge_history.jsonl")
        history = get_ingestion_history(history_file)
        
        if history:
            # Convert to dataframe for nicer display
            st.dataframe(history, use_container_width=True)
        else:
            st.write("_(Sin historial previo)_")

    except ImportError:
        st.warning("No se pudo importar `get_stats` de `knowledge.vector`. Revisa la instalación.")
    except Exception as e:
        st.warning(f"No se pudo leer la DB: {e}")

    # -------------------------------------------------------------------------
    # 🛠 Tools Manager Tab
    # -------------------------------------------------------------------------
    with tab_tm:
        st.markdown("### 🛠 Gestor de Herramientas")
        
        from agnostic_agent.tools import TOOL_REGISTRY
        
        # Initialize session state for tools config if not exists
        if "tools_config" not in st.session_state:
            # User request: Default to ALL ENABLED (True) to allow knowledge queries out-of-the-box
            st.session_state.tools_config = {name: True for name in TOOL_REGISTRY.keys()}

        if "inspect_tool_name" not in st.session_state:
             st.session_state.inspect_tool_name = None

        def _reset_agent():
            st.session_state.agent = None
            st.toast("Configuración de herramientas modificada. Agente reiniciado.", icon="🛠")
        
        # --- Tool Toggles (Grouped) ---
        st.markdown("#### Configuración")
        
        # Define groups
        tool_groups = {
            "🛠 Básicas": ["to_upper", "word_count", "is_palindrome"],
            "🧮 Matemáticas": ["eval_math_expression", "sum_numbers", "average_numbers"],
            "🧠 RAG / Knowledge": ["search_knowledge_base", "semantic_search", "embed_texts", "rerank_qwen3"],
            "📊 Contexto Tabular": ["semantic_search_in_csv", "embed_context_tables", "judge_row_with_context"],
        }
        
        # Tools not in any group (fallback)
        all_tools = set(TOOL_REGISTRY.keys())
        grouped_tools = set()
        for g_list in tool_groups.values():
            grouped_tools.update(g_list)
        others = list(all_tools - grouped_tools)
        if others:
            tool_groups["🔧 Otras"] = others

        # Render groups
        for group_name, tools_in_group in tool_groups.items():
            valid_tools = [t for t in tools_in_group if t in TOOL_REGISTRY]
            if not valid_tools:
                continue
                
            st.markdown(f"**{group_name}**")
            cols = st.columns(3)
            for i, tool_name in enumerate(valid_tools):
                col = cols[i % 3]
                is_active = st.session_state.tools_config.get(tool_name, True)
                
                # Split layout: Toggle (State) | Button (Inspect)
                c_toggle, c_name = col.columns([0.25, 0.75])
                
                with c_toggle:
                    new_state = st.toggle(
                        label="On/Off", 
                        value=is_active, 
                        key=f"toggle_{tool_name}",
                        on_change=_reset_agent,
                        label_visibility="collapsed"
                    )
                    st.session_state.tools_config[tool_name] = new_state
                
                with c_name:
                    if st.button(tool_name, key=f"btn_inspect_{tool_name}", use_container_width=True):
                        st.session_state.inspect_tool_name = tool_name
            
            st.caption("") # Spacer

        st.divider()

        # Re-map tools for the inspector based on current config (or all if we want to inspect disabled ones too?)
        # Let's inspect 'enabled' ones or just all from registry? 
        # For consistency with the old logic:
        _tools_list = list(TOOL_REGISTRY.values()) # All available for inspection
        tools_map = {t.name: t for t in _tools_list}

        # --- Tool Inspection Section ---
        st.markdown("### 🔬 Inspector de Herramientas")
        
        # Select a tool to inspect
        tool_names = sorted(list(tools_map.keys()))
        
        # Determine index for selectbox based on session state
        sb_index = 0
        if st.session_state.inspect_tool_name in tool_names:
            sb_index = tool_names.index(st.session_state.inspect_tool_name)

        if tool_names:
            def _update_inspect_tool():
                st.session_state.inspect_tool_name = st.session_state.sb_inspect_tool

            selected_tool_name = st.selectbox(
                "Selecciona una herramienta para inspeccionar:", 
                tool_names,
                index=sb_index,
                key="sb_inspect_tool",
                on_change=_update_inspect_tool
            )
            # Sync session state if changed manually via selectbox (though on_change handles it, 
            # we ensure consistency if re-run happens elsewhere)
            st.session_state.inspect_tool_name = selected_tool_name
            
            if selected_tool_name:
                tool = tools_map[selected_tool_name]
                
                c1, c2 = st.columns([1, 1])
                
                with c1:
                    st.markdown(f"**Nombre:** `{tool.name}`")
                    st.markdown("**Descripción:**")
                    st.info(tool.description or "Sin descripción")
                    
                    # Show Input Schema cleanly
                    st.markdown("**Esquema de Entrada (Args):**")
                    if tool.args_schema:
                        try:
                            schema = tool.args_schema.schema()
                            props = schema.get("properties", {})
                            required = schema.get("required", [])
                            
                            # Render as a markdown table or list
                            if not props:
                                st.markdown("_(Sin argumentos)_")
                            else:
                                for prop_name, prop_info in props.items():
                                    is_req = "*(obligatorio)*" if prop_name in required else ""
                                    t_type = prop_info.get("type", "any")
                                    desc = prop_info.get("description", "")
                                    st.markdown(f"- **`{prop_name}`** `({t_type})` {is_req}")
                                    if desc:
                                        st.markdown(f"  > {desc}")
                        except Exception as e:
                            st.error(f"Error generando esquema: {e}")
                            st.json(tool.args_schema.schema()) # Fallback
                    else:
                        st.text("Sin esquema definido (str por defecto)")

                with c2:
                    st.markdown("**🕸 Visualización del Proceso**")
                    st.graphviz_chart(f'''
                        digraph ToolProc {{
                            rankdir=LR;
                            node [shape=box, style=filled, color=lightblue];
                            
                            In [label="Input", shape=ellipse, color=lightgrey];
                            Proc [label="{tool.name}", shape=box, color=orange, style="rounded,filled"];
                            Out [label="Output", shape=ellipse, color=lightgreen];
                            
                            In -> Proc [label="args"];
                            Proc -> Out [label="return"];
                        }}
                    ''')
        if not tool_names:
            st.warning("No hay herramientas registradas.")

    # -------------------------------------------------------------------------
    # 🧩 Skills Manager Tab
    # -------------------------------------------------------------------------
    with tab_skills:
        st.markdown("### 🧩 Gestor de Skills")
        st.info("Las **Skills** son recetas avanzadas que combinan herramientas y conocimiento para resolver tareas complejas.")
        
        if agent and agent.skill_registry:
            skills = agent.skill_registry.list_skills(enabled_only=False)
            if not skills:
                st.warning("No se encontraron skills en la carpeta `skills/`.")
            else:
                for skill in skills:
                    c1, c2 = st.columns([0.8, 0.2])
                    with c1:
                        st.markdown(f"**{skill.name}**")
                        st.caption(skill.description)
                    with c2:
                        # Toggle
                        is_on = st.toggle("Activar", value=skill.enabled, key=f"skill_toggle_{skill.name}", label_visibility="collapsed")
                        
                        if is_on != skill.enabled:
                            agent.skill_registry.set_enabled(skill.name, is_on)
                            # Persist
                            if "skills_config" not in st.session_state:
                                st.session_state.skills_config = {}
                            st.session_state.skills_config[skill.name] = is_on
                            st.rerun()

                    with st.expander(f"Ver receta: {skill.name}"):
                        st.markdown(f"**Tools requeridas:** `{skill.tools}`")
                        st.markdown(f"**Knowledge requerida:** `{skill.knowledge}`")
                        st.markdown("---")
                        st.markdown(skill.instructions)
                        if skill.file_path:
                            st.caption(f"Fuente: `{skill.file_path}`")
                    
                    st.divider()
        else:
            st.error("No se ha cargado el registro de skills (Agent no inicializado o sin registry).")
    # Removed old Agent Config section (merged to top sidebar)
    
    # Config moved to get_or_init_agent
    pass
    # -------------------------------------------------------------------------
    # 📜 Logs de Ejecución Tab
    # -------------------------------------------------------------------------
    with tab_logs:
        st.markdown("### 📜 Logs de Ejecución (Tiempo Real)")
        
        if "tool_logs" not in st.session_state:
            st.session_state["tool_logs"] = []
            
        logs = st.session_state["tool_logs"]
        
        if logs:
            c_clear, c_spacer = st.columns([0.2, 0.8])
            with c_clear:
                if st.button("🗑️ Limpiar Logs", key="clear_logs_tab"):
                     st.session_state["tool_logs"] = []
                     st.rerun()

            # Reverse to show newest first
            for i, log_entry in enumerate(reversed(logs)):
                # Unique key for expander
                ts = log_entry.get('timestamp', '?')
                tname = log_entry.get('tool', 'unknown')
                
                with st.expander(f"⏰ {ts} | 🛠 {tname}", expanded=(i==0)):
                    st.markdown("**Input:**")
                    st.code(json.dumps(log_entry.get('input',{}), indent=2, ensure_ascii=False), language="json")
                    
                    st.markdown("**Output:**")
                    out_val = log_entry.get('output', '')
                    if isinstance(out_val, (dict, list)):
                        st.code(json.dumps(out_val, indent=2, ensure_ascii=False), language="json")
                    else:
                        st.code(str(out_val))
        else:
            st.info("No hay logs de ejecución recientes.")

