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
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
.block-container { padding-top: 1rem !important; }
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
  padding: 12px;
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
    st.markdown("## 🧪 Chat Studio")
    agent_mode = st.selectbox(
        "Policy mode",
        ["tools_strict", "free_policies"],
        index=0 if st.session_state.agent_mode == "tools_strict" else 1,
    )

    st.markdown("### 🧭 Inspector")
    show_thinking_tab = st.checkbox("🧠 Thinking", value=True)
    show_deep_tab = st.checkbox("🧠 Deep", value=True)
    show_dev_tab = st.checkbox("🔍 Dev", value=True)

    st.markdown("### 🧹 Acciones")
    cA, cB = st.columns(2)
    with cA:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent = None
            st.session_state.selected_msg_id = None
            st.toast("Chat reiniciado.", icon="🧹")
            st.rerun()
    with cB:
        if st.button("⬇️ Export", use_container_width=True):
            export = {"agent_mode": st.session_state.agent_mode, "messages": st.session_state.messages}
            st.session_state.export_json = json.dumps(export, ensure_ascii=False, indent=2)
            st.toast("Transcript listo.", icon="⬇️")

    if isinstance(st.session_state.export_json, str):
        st.download_button(
            "Descargar transcript.json",
            data=st.session_state.export_json,
            file_name="transcript.json",
            mime="application/json",
            use_container_width=True,
        )

# Mode change => reset agent (history stays)
if st.session_state.agent_mode != agent_mode:
    st.session_state.agent_mode = agent_mode
    st.session_state.agent = None
    st.toast(f"Modo: {agent_mode}", icon="🧭")

# -----------------------------
# Helpers
# -----------------------------
def next_id() -> int:
    st.session_state.msg_counter += 1
    return st.session_state.msg_counter

def get_or_init_agent(mode: str) -> Agent:
    if st.session_state.agent is None:
        # Actualizamos envs por si acaso, aunque Agent.init(PlannerConfig(...)) es más explícito
        os.environ["PLANNER_POLICY_MODE"] = mode
        os.environ["AGENT_POLICY_MODE"] = mode
        
        with st.spinner(f"Inicializando agente (mode={mode})…"):
            try:
                # CREAMOS CONFIG DE PLANNER EXPLÍCITA
                cfg = PlannerConfig(policy_mode=mode)
                
                # Inicializamos pasando esa config
                st.session_state.agent = Agent.init(config_or_setup=cfg)
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
mode_badge = "tools_strict" if st.session_state.agent_mode == "tools_strict" else "free_policies"
mode_class = "accent" if st.session_state.agent_mode == "tools_strict" else "good"

st.markdown(
    f"""
<div class="topbar">
  <div class="brand">
    <div class="logo">🧪</div>
    <div>
      <div class="title">Agnostic Agent · Chat Studio</div>
      <div class="subtitle">Feed limpio + Inspector dinámico (thinking/deep/dev)</div>
    </div>
  </div>
  <div class="badges">
    <span class="badge {mode_class}">🧭 {mode_badge}</span>
    <span class="badge">🔎 inspector: on</span>
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
                    st.markdown(f'<div class="bubble-user">{html.escape(msg.get("content",""))}</div>', unsafe_allow_html=True)

            elif role == "assistant":
                out = msg.get("out") or {}
                answer = strip_user_prefix(as_text(out.get("user_out")))
                raw_state = get_raw_state(out)
                tool_runs = extract_tool_runs(out, raw_state)

                # --- Capture Tool Logs for Offline Manager ---
                if "tool_logs" not in st.session_state:
                    st.session_state["tool_logs"] = []
                
                import datetime
                for tr in tool_runs:
                    # Check if this run is already logged to avoid duplicates? 
                    # Simpler is to just append, but we might get duplicates on re-runs.
                    # Ideally, logging should happen inside the agent, but here we extract for display.
                    # We'll prepend to be newer first, or append and show reversed.
                    
                    # Using a simple signature to avoid dupes in this session view if needed, 
                    # but for now let's just add them.
                    
                    log_entry = {
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "tool": tr.get("name", "unknown"),
                        "input": tr.get("args", {}),
                        "output": tr.get("output", "")
                    }
                    st.session_state["tool_logs"].append(log_entry)
                # ---------------------------------------------

                with st.chat_message("assistant"):
                    # Pretty answer only
                    card_md(
                        title="Respuesta",
                        body_md=html.escape(answer or "_(sin respuesta)_").replace("\n", "<br>"),
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
        st.markdown('<div class="inspector">', unsafe_allow_html=True)
        st.markdown("### 🔎 Inspector")

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
                        body = deep_txt or "_(vacío)_"
                        body_html = html.escape(body).replace("\n", "<br>")
                        card_md("Vista profunda (deep_out / summary)", body_html, icon="🧠", hint="pipeline")

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
        st.session_state.messages.append({"id": uid, "role": "user", "content": prompt, "out": None})

        agent = get_or_init_agent(st.session_state.agent_mode)
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
    # Prepare tool list
    # If we have a tools_config in session state, filter the tools
    if "tools_config" in st.session_state:
        # Prepare tool list
        # If we have a tools_config in session state, filter the tools
        if "tools_config" in st.session_state:
            enabled_tools = [name for name, active in st.session_state.tools_config.items() if active]
        else:
            enabled_tools = None # All tools

        tools_map = get_default_tools(enabled_tools)
        
        # --- Tool Inspection Section ---
        st.divider()
        st.markdown("### 🔬 Inspector de Herramientas")
        
        # Select a tool to inspect
        tool_names = list(tools_map.keys())
        if tool_names:
            selected_tool_name = st.selectbox("Selecciona una herramienta para inspeccionar:", tool_names)
            
            if selected_tool_name:
                tool = tools_map[selected_tool_name]
                
                c1, c2 = st.columns([1, 1])
                
                with c1:
                    st.markdown(f"**Nombre:** `{tool.name}`")
                    st.markdown("**Descripción:**")
                    st.info(tool.description or "Sin descripción")
                    
                    # Show Input Schema if pydantic args
                    st.markdown("**Esquema de Entrada (Args):**")
                    if tool.args_schema:
                        st.json(tool.args_schema.schema())
                    else:
                        st.text("Sin esquema definido (str por defecto)")

                with c2:
                    st.markdown("**🕸 Visualización del Proceso**")
                    # Graphviz visualization of this specific tool's flow
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

        else:
            st.warning("No hay herramientas habilitadas.")

        st.divider()
        
        # --- Real-time Logs Visualization ---
        st.markdown("### 📜 Logs de Ejecución (Tiempo Real)")
        
        # Initialize logs in session state if not present
        if "tool_logs" not in st.session_state:
            st.session_state["tool_logs"] = []
            
        # Display logs
        if st.session_state["tool_logs"]:
            for log_entry in reversed(st.session_state["tool_logs"][-10:]): # Show last 10
                with st.chat_message("tool", avatar="🛠"):
                    st.markdown(f"**{log_entry['timestamp']}** - `{log_entry['tool']}`")
                    with st.expander("Ver detalles"):
                        st.markdown("**Input:**")
                        st.code(str(log_entry['input']))
                        st.markdown("**Output:**")
                        st.code(str(log_entry['output']))
        else:
            st.caption("No hay logs de ejecución recientes.")

    # ---------------------------------------------------------------------
    
    # Instantiate agent with logging wrapper if we want to capture logs (TODO: Implement callback/wrapper)
    # For now, we will just use the standard instantiation but we need to hook into it for logging.
    # Since we can't easily hook without changing Agent class, we'll rely on the Agent's output 
    # to populate 'tool_logs' in the processing loop (tab_online).
    
    agent = get_or_init_agent(st.session_state.agent_mode)
    
    # If we want to apply the tools_config dynamically:
    if "tools_config" in st.session_state:
        enabled_tools = [name for name, active in st.session_state.tools_config.items() if active]
        agent.tools = get_default_tools(enabled_tools)
        agent.tools_map = {t.name: t for t in agent.tools}
    # Create sub-tabs
    tab_km, tab_tm = st.tabs(["📚 Knowledge Manager", "🛠 Tools Manager"])

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
                from agnostic_agent.knowledge_offline import ingest_pdf_file
                
                with st.spinner("Procesando documento... (puede tardar si usa CPU)"):
                    try:
                        res = ingest_pdf_file(save_path, DB_PATH)
                        if "error" in res:
                            st.error(f"Error: {res['error']}")
                            st.info("Nota: Si 'Docling' o 'PyMuPDF' no están instalados, no se podrá extraer texto.")
                        else:
                            st.balloons()
                            st.success("¡Ingesta completada!")
                            st.json(res)
                    except Exception as e:
                        st.error(f"Excepción crítica: {e}")

        st.divider()
        
        # DB Stats
        st.markdown("### 📊 Estado de la Base de Datos")
        try:
            from agnostic_agent.knowledge_offline import get_stats
            stats = get_stats(DB_PATH)
            c1, c2 = st.columns(2)
            c1.metric("Chunks Indexados", stats.get("chunks", 0))
            c2.metric("Archivos Únicos", stats.get("files", 0))
        except ImportError:
            st.warning("No se pudo importar `get_stats` de `knowledge_offline`. Revisa la instalación.")
        except Exception as e:
            st.warning(f"No se pudo leer la DB: {e}")

    # -------------------------------------------------------------------------
    # 🛠 Tools Manager Tab
    # -------------------------------------------------------------------------
    with tab_tm:
        st.markdown("### 🛠 Gestor de Herramientas")
        st.markdown("Activa o desactiva las herramientas disponibles para el agente.")
        
        from agnostic_agent.tools import TOOL_REGISTRY
        
        # Initialize session state for tools config if not exists
        if "tools_config" not in st.session_state:
            st.session_state.tools_config = {name: True for name in TOOL_REGISTRY.keys()}
        
        # --- Tool Toggles ---
        cols = st.columns(3)
        for i, tool_name in enumerate(TOOL_REGISTRY.keys()):
            col = cols[i % 3]
            is_active = st.session_state.tools_config.get(tool_name, True)
            
            new_state = col.toggle(
                label=tool_name, 
                value=is_active, 
                key=f"toggle_{tool_name}"
            )
            st.session_state.tools_config[tool_name] = new_state

        st.divider()
        
        # --- Visualization (Dummy for now, could be real history) ---
        st.markdown("### 🕸 Visualización de Procesos (Input → Output)")
        
        # Example visualization using graphviz or mermaid
        # For now, let's show a simple example of how a tool flow looks
        
        st.graphviz_chart('''
            digraph ToolsFlow {
                rankdir=LR;
                node [shape=box, style=filled, color=lightblue];
                
                Input [label="Input User Query", shape=ellipse, color=lightgrey];
                Agent [label="Agent Decision", shape=diamond, color=orange];
                
                Input -> Agent;
                
                subgraph cluster_tools {
                    label = "Active Tools";
                    style=dashed;
                    color=grey;
                    
                    Tool1 [label="search_knowledge_base"];
                    Tool2 [label="sum_numbers"];
                }
                
                Agent -> Tool1 [label="call"];
                Tool1 -> Output1 [label="result"];
                
                Output1 [label="Output Context", shape=ellipse, color=lightgreen];
            }
        ''')
        
        with st.expander("Ver logs de ejecución recientes (Simulado)"):
            st.code("""
[10:00:01] Agent called 'search_knowledge_base' with query="OpenAI Gym"
[10:00:02] Tool 'search_knowledge_base' processing...
[10:00:03] Tool returned 3 chunks.
            """, language="log")
