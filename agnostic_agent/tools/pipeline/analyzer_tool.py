from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _guess_response_mode(user_prompt: str, subqueries: List[str]) -> str:
    text = (user_prompt or "").lower()
    if len(subqueries) > 1:
        return "batch_summary"
    if any(tok in text for tok in ["audita", "auditar", "drift", "valida", "validar", "revisa"]):
        return "audit"
    if any(tok in text for tok in ["json", "tecnico", "técnico", "schema", "dag", "sql"]):
        return "technical"
    return "user"


def _infer_intents_for_subquery(subquery: str, selected_skill: str) -> List[str]:
    text = (subquery or "").lower()
    skill = (selected_skill or "").lower()
    if skill == "semantic_researcher":
        if any(tok in text for tok in ["resumen", "resume", "sintetiza", "conclusion"]):
            return ["semantic_synthesis"]
        return ["semantic_lookup"]
    if skill == "contabilidad_automatica":
        if any(tok in text for tok in ["drift", "descuadre", "concili", "cuadr"]):
            return ["reconcile_credit"]
        if any(tok in text for tok in ["regla", "tasa", "saneamiento"]):
            return ["explain_rule"]
        return ["query_financial_data"]
    if skill == "chat_db":
        if any(tok in text for tok in ["schema", "tabla", "columna", "estructura"]):
            return ["explain_schema"]
        if len(re.findall(r"\{[^{}]+\}", subquery or "")) > 1:
            return ["batch_query"]
        if any(tok in text for tok in ["compar", "vs", "contra"]):
            return ["compare_entities"]
        if any(tok in text for tok in ["cuanto", "cuánt", "count", "sum", "avg", "promedio", "max", "min"]):
            return ["aggregate_data"]
        return ["query_data"]
    return ["general_query"]


def _extract_entities(subquery: str) -> Dict[str, Any]:
    text = subquery or ""
    entities: Dict[str, Any] = {}
    raw_jsons = re.findall(r"\{[^{}]+\}", text)
    for chunk in raw_jsons:
        try:
            parsed = json.loads(chunk)
        except Exception:
            continue
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if value not in (None, ""):
                    entities[str(key)] = value
    loc_match = re.search(r"\bLOC-\d{3,}\b", text, flags=re.IGNORECASE)
    if loc_match and "credito_id" not in entities:
        entities["credito_id"] = loc_match.group(0).upper()
    db_matches = re.findall(r"\b([A-Za-z0-9_.-]+\.db)\b", text, flags=re.IGNORECASE)
    if db_matches:
        entities["db_files"] = db_matches
    return entities


def _derive_constraints(subquery: str, entities: Dict[str, Any], intents: List[str]) -> Dict[str, Any]:
    text = (subquery or "").lower()
    constraints: Dict[str, Any] = {}
    if "batch_query" in intents:
        constraints["batch"] = True
    if "credito_id" in entities:
        constraints["entity_scope"] = "single_credit"
    if any(tok in text for tok in ["solo ", "únicamente", "unicamente", "only "]):
        constraints["strict_filtering"] = True
    if any(tok in text for tok in ["ultim", "reciente", "latest", "top", "primer"]):
        constraints["ordering_required"] = True
    return constraints


def execute_analyzer_tool(
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
                    "world": getattr(s, "world", s.name),
                    "tools": list(s.tools or []),
                    "knowledge": list(s.knowledge or []),
                    "intents": list(getattr(s, "intents", []) or []),
                    "entities": list(getattr(s, "entities", []) or []),
                    "planner": dict(getattr(s, "planner_policy", {}) or {}),
                    "summarizer": dict(getattr(s, "summarizer_policy", {}) or {}),
                    "validator": dict(getattr(s, "validator_policy", {}) or {}),
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
    selected_skill_world = ""
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
        selected_skill_world = str(data.get("selected_skill_world", "")).strip()

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
        selected_skill_world = selected_skills[0]

    if not selected_skill_world and selected_skills:
        selected_skill_world = selected_skills[0]
    if not selected_skill_world and normalized_allowlist:
        selected_skill_world = normalized_allowlist[0]
    if not selected_skill_world and skill_registry:
        preferred = None
        user_text_low = (user_prompt or "").lower()
        if any(tok in user_text_low for tok in ["credito", "crédito", "saneamiento", "saldo", "contabilidad", "loc-"]):
            preferred = "contabilidad_automatica"
        elif any(tok in user_text_low for tok in ["documento", "fuente", "pdf", "investiga", "semantic", "chunk"]):
            preferred = "semantic_researcher"
        elif any(tok in user_text_low for tok in ["db", "tabla", "sql", "sqlite", "columna", "registro"]):
            preferred = "chat_db"
        if preferred and skill_registry.get_world(preferred):
            selected_skill_world = preferred
            selected_skills = [skill_registry.get_world(preferred).name]
    if not selected_skills and selected_skill_world and skill_registry:
        skill_obj = skill_registry.get_world(selected_skill_world)
        if skill_obj:
            selected_skills = [skill_obj.name]

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
    active_skill = selected_skills[0] if selected_skills else selected_skill_world
    subquery_intents = [_infer_intents_for_subquery(sq, active_skill) for sq in subqueries]
    entities_by_subquery = [_extract_entities(sq) for sq in subqueries]
    constraints_by_subquery = [
        _derive_constraints(sq, entities, intents)
        for sq, entities, intents in zip(subqueries, entities_by_subquery, subquery_intents)
    ]
    response_mode = _guess_response_mode(user_prompt, subqueries)
    selection_mode = "forced" if normalized_allowlist else "auto"
    world_contract: Dict[str, Any] = {}
    if skill_registry and active_skill:
        skill_obj = skill_registry.get_world(active_skill)
        if skill_obj:
            world_contract = {
                "name": skill_obj.name,
                "world": skill_obj.world or skill_obj.name,
                "description": skill_obj.description,
                "tools": list(skill_obj.tools or []),
                "knowledge": list(skill_obj.knowledge or []),
                "intents": list(skill_obj.intents or []),
                "entities": list(skill_obj.entities or []),
                "planner": dict(skill_obj.planner_policy or {}),
                "summarizer": dict(skill_obj.summarizer_policy or {}),
                "validator": dict(skill_obj.validator_policy or {}),
                "ui": dict(skill_obj.ui or {}),
            }
    analyzer = {
        "input_payload": {"user_prompt": user_prompt},
        "selected_skill_world": selected_skill_world,
        "selection_mode": selection_mode,
        "propositional_logic": logic_form,
        "subqueries": subqueries,
        "subqueries_logic": subqueries_logic,
        "subquery_intents": subquery_intents,
        "entities_by_subquery": entities_by_subquery,
        "constraints_by_subquery": constraints_by_subquery,
        "response_mode": response_mode,
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
        "selected_skill_world": selected_skill_world,
        "subquery_intents": subquery_intents,
        "entities_by_subquery": entities_by_subquery,
        "constraints_by_subquery": constraints_by_subquery,
        "world_contract": world_contract,
        "messages": [analyzer_msg],
    }

