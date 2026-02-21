import streamlit as st
from typing import List, Tuple

from agnostic_agent.ui.panels.helpers import (
    assistant_messages,
    as_text,
    card_code,
    card_md,
    extract_summary_deep,
    extract_thinking,
    extract_tool_runs,
    find_message_by_id,
    get_raw_state,
    render_tool_runs,
    strip_user_prefix,
)


def render_inspector(show_title: bool = True, boxed: bool = True):
    if boxed:
        root_ctx = st.container(border=True)
    else:
        root_ctx = st.container()

    with root_ctx:
        if show_title:
            st.markdown("### Inspector")

        a_msgs = assistant_messages()
        if not a_msgs:
            st.info("Aun no hay respuestas del agente. Escribe algo para empezar.")
            return

        ids = [m["id"] for m in a_msgs]

        def label(mid: int) -> str:
            m = find_message_by_id(mid) or {}
            out = m.get("out") or {}
            text = strip_user_prefix(as_text(out.get("user_out"))).replace("\n", " ").strip()
            text = (text[:60] + "...") if len(text) > 60 else text
            return f"id={mid} - {text or '(sin texto)'}"

        if st.session_state.selected_msg_id not in ids:
            st.session_state.selected_msg_id = ids[-1]

        idx = 0
        try:
            idx = ids.index(st.session_state.selected_msg_id)
        except ValueError:
            pass

        sel = st.selectbox(
            "Mensaje seleccionado",
            options=ids,
            index=idx,
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
        if st.session_state.get("show_thinking_tab", True):
            tab_specs.append(("Thinking", "thinking"))
        if st.session_state.get("show_deep_tab", True):
            tab_specs.append(("Deep", "deep"))
        if st.session_state.get("show_dev_tab", True):
            tab_specs.append(("Dev", "dev"))

        if not tab_specs:
            return

        tabs = st.tabs([t[0] for t in tab_specs])
        for (_, tab_key), tab in zip(tab_specs, tabs):
            with tab:
                if tab_key == "thinking":
                    card_code(
                        "Pensamiento (thinking)",
                        thinking,
                        icon="mind",
                        hint="reasoning_content",
                    )
                elif tab_key == "deep":
                    content_to_show = deep_txt if deep_txt else "_(vacio / sin resumen)_"
                    card_md(
                        "Vista profunda (deep_out / summary)",
                        content_to_show,
                        icon="deep",
                        hint="pipeline",
                    )
                elif tab_key == "dev":
                    render_tool_runs(tool_runs)
                    with st.expander("raw_state (debug)", expanded=False):
                        if isinstance(raw_state, dict) and raw_state:
                            st.json(raw_state)
                        else:
                            st.markdown("_(sin raw_state)_")
