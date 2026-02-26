from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def execute_analyzer_node(
    state: Dict[str, Any],
    *,
    tools: List[Any],
    cfg: Any,
    planner_llm: Any,
    skill_registry: Any,
    ai_message_type: Any,
    human_message_type: Any,
    system_message_type: Any,
    coerce_content_str: Any,
    sanitize_subquery_text: Any,
    extract_top_level_json_objects: Any,
    is_placeholder_subquery: Any,
) -> Dict[str, Any]:
    messages = state.get("messages", [])
    user_messages = [m for m in messages if isinstance(m, human_message_type)]
    last_user = user_messages[-1] if user_messages else None
    user_text = last_user.content if isinstance(last_user, human_message_type) else ""
    user_prompt = state.get("user_prompt") or user_text
    active_tools_input = tools
    knowledge_names = state.get("knowledge_names", [])
    knowledge_available = bool(knowledge_names)

    from agnostic_agent.prompts import ANALYZER_SYSTEM_PROMPT, LOGIC_DEFINITIONS

    def _instruction_summary(instructions: str, max_len: int = 260) -> str:
        if not instructions:
            return ""
        lines: List[str] = []
        for ln in instructions.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("#"):
                continue
            lines.append(ln)
            if len(" ".join(lines)) >= max_len:
                break
        txt = " ".join(lines)
        return (txt[:max_len] + "...") if len(txt) > max_len else txt

    def _build_skills_catalog() -> List[Dict[str, Any]]:
        if not skill_registry:
            return []
        catalog: List[Dict[str, Any]] = []
        for s in skill_registry.list_skills():
            catalog.append(
                {
                    "name": s.name,
                    "description": s.description or "",
                    "tools": list(s.tools or []),
                    "knowledge": list(s.knowledge or []),
                    "summary": _instruction_summary(getattr(s, "instructions", "")),
                }
            )
        return catalog

    skills_catalog = _build_skills_catalog()
    available_skills_txt = (
        json.dumps(skills_catalog, ensure_ascii=False, indent=2) if skills_catalog else "[]"
    )
    sys_content = (
        ANALYZER_SYSTEM_PROMPT.replace("{user_prompt}", user_prompt)
        .replace("{knowledge_available}", str(knowledge_available))
        .replace("{knowledge_names}", str(knowledge_names))
        .replace("{LOGIC_DEFINITIONS}", LOGIC_DEFINITIONS)
        .replace("{AVAILABLE_SKILLS}", available_skills_txt or "[]")
    )
    if skills_catalog:
        sys_content += f"\n\nSKILLS DISPONIBLES (CATALOGO ESTRUCTURADO):\n{available_skills_txt}"
    if cfg and not cfg.enable_thinking:
        sys_content += "\n\nCRITICAL: DO NOT use <think> tags. Respond ONLY with the JSON block."

    sys_msg = system_message_type(content=sys_content)
    user_msg = human_message_type(content="Analiza mi peticion y genera el JSON.")

    selected_skills: List[str] = []
    analyzer_skill_selection: Dict[str, Any] = {
        "source": "llm",
        "selected_skill": None,
        "score": None,
    }
    response: Optional[Any] = None
    subqueries = [user_prompt]
    logic_form = "q1"

    forced_skill = state.get("forced_skill")
    skills_allowlist = state.get("skills_allowlist") or []
    if forced_skill and forced_skill != "Auto (Analyzer)":
        skills_allowlist = [forced_skill]

    normalized_allowlist: List[str] = []
    if skills_allowlist:
        for s in skills_allowlist:
            if not s or s == "Auto (Analyzer)":
                continue
            if skill_registry is None or skill_registry.get_skill(s):
                normalized_allowlist.append(s)

    try:
        response = planner_llm.invoke([sys_msg, user_msg])
        raw_content = getattr(response, "content", "")
        if isinstance(raw_content, dict):
            data = raw_content
            content = ""
        else:
            content = coerce_content_str(raw_content)
            data = None

        if data is None and "```" in content:
            import re

            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*", "", content)

        if data is None:
            content = content.strip()
            if not content.startswith("{") and "{" in content:
                content = content[content.find("{") :]
                if "}" in content:
                    content = content[: content.rfind("}") + 1]
            data = json.loads(content)

        subqueries = data.get("subqueries", [user_prompt])
        if isinstance(subqueries, str):
            subqueries = [subqueries]
        elif not isinstance(subqueries, list):
            subqueries = [subqueries]
        logic_form = data.get("logic_form", "q1")
        selected_skills = data.get("selected_skills", [])

        if normalized_allowlist:
            selected_skills = [s for s in selected_skills if s in normalized_allowlist]
            if not selected_skills:
                selected_skills = list(normalized_allowlist)
            analyzer_skill_selection = {
                "source": "allowlist",
                "selected_skill": selected_skills[0],
                "score": None,
            }
        elif selected_skills and analyzer_skill_selection.get("selected_skill") is None:
            analyzer_skill_selection = {
                "source": "llm",
                "selected_skill": selected_skills[0],
                "score": None,
            }

        logger.debug("analyzer parsed json successfully; selected_skills=%s", selected_skills)
    except Exception as e:
        response_content = getattr(response, "content", "") if response is not None else ""
        logger.warning(
            "analyzer failed parsing json; err=%r content_preview=%s",
            e,
            str(response_content)[:100],
        )
        if knowledge_available:
            logger.info("analyzer fallback activated: leaving skills empty")
            selected_skills = []

    if normalized_allowlist:
        selected_skills = [s for s in selected_skills if s in normalized_allowlist]
        if not selected_skills:
            selected_skills = list(normalized_allowlist)
        analyzer_skill_selection = {
            "source": "allowlist",
            "selected_skill": selected_skills[0],
            "score": None,
        }

    subqueries = [sq for sq in (sanitize_subquery_text(s) for s in subqueries) if sq]
    if len(subqueries) == 1:
        prompt_text = coerce_content_str(user_prompt)
        object_chunks = extract_top_level_json_objects(prompt_text)
        if len(object_chunks) > 1:
            prefix = prompt_text.split("{", 1)[0].strip()
            if prefix:
                subqueries = [f"{prefix} {obj}".strip() for obj in object_chunks]
            else:
                subqueries = object_chunks
    if subqueries and all(is_placeholder_subquery(s) for s in subqueries):
        subqueries = [sanitize_subquery_text(user_prompt) or user_prompt]
    if not subqueries:
        subqueries = [sanitize_subquery_text(user_prompt) or user_prompt]
    logic_form = " AND ".join(f"q{i+1}" for i in range(len(subqueries)))

    subqueries_logic = [f"q{i+1}" for i in range(len(subqueries))]
    analyzer = {
        "input_payload": {"user_prompt": user_prompt},
        "propositional_logic": logic_form,
        "subqueries": subqueries,
        "subqueries_logic": subqueries_logic,
    }

    analyzer_msg = ai_message_type(
        content=f"### ANALYZER (JSON Mode)\nSkills: {selected_skills}\nSubqueries: {subqueries}",
        additional_kwargs={"pipeline_internal": True, "node": "analyzer"},
    )

    _ = active_tools_input
    return {
        "analyzer": analyzer,
        "_active_skills_internal": selected_skills,
        "_analyzer_skill_selection": analyzer_skill_selection,
        "messages": [analyzer_msg],
    }
