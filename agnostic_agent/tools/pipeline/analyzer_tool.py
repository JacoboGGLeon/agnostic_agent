from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_GENERIC_ID_TOKEN_RE = re.compile(r"\b[A-Za-z]{2,}[A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b")
_PLURAL_BATCH_HINTS = (
    "siguientes",
    "estos",
    "estas",
    "listados",
    "listadas",
    "todos",
    "todas",
    "varios",
    "varias",
    "multiple",
    "múltiple",
    "multiples",
    "múltiples",
)
_REFERENTIAL_HINTS = (
    "estos",
    "estas",
    "esos",
    "esas",
    "aquellos",
    "aquellas",
    "listados",
    "listadas",
    "anteriores",
    "mencionados",
    "mencionadas",
    "mismos",
    "mismas",
    "detalle",
    "dicha",
    "dicho",
    "esa",
    "ese",
    "esta",
    "este",
    "esto",
    "resultado",
    "conciliacion",
    "conciliación",
    "flujos",
)
_REFERENTIAL_PRONOUN_RE = re.compile(r"\b\w+(?:los|las)\b", flags=re.IGNORECASE)
_SINGULAR_REFERENTIAL_HINTS = (
    "dicha",
    "dicho",
    "esa conciliacion",
    "esa conciliación",
    "este resultado",
    "ese resultado",
    "esta conciliacion",
    "esta conciliación",
    "como llegaste",
    "cómo llegaste",
    "detallame",
    "detállame",
    "flujos",
    "esto",
)


def _guess_response_mode(user_prompt: str, subqueries: List[str]) -> str:
    text = (user_prompt or "").lower()
    if len(subqueries) > 1:
        return "batch_summary"
    if any(tok in text for tok in ["audita", "auditar", "drift", "valida", "validar", "revisa"]):
        return "audit"
    if any(tok in text for tok in ["json", "tecnico", "técnico", "schema", "dag", "sql"]):
        return "technical"
    return "user"


def _contains_any(text: str, tokens: List[str]) -> bool:
    haystack = (text or "").lower()
    return any(token in haystack for token in tokens)


def _build_world_contract(skill_registry: Any, skill_name: str) -> Dict[str, Any]:
    if not skill_registry or not skill_name:
        return {}
    skill_obj = skill_registry.get_world(skill_name)
    if skill_obj is None and hasattr(skill_registry, "get_skill"):
        skill_obj = skill_registry.get_skill(skill_name)
    if skill_obj is None:
        return {}
    return {
        "name": skill_obj.name,
        "world": skill_obj.world or skill_obj.name,
        "description": skill_obj.description,
        "tools": list(skill_obj.tools or []),
        "knowledge": list(skill_obj.knowledge or []),
        "intents": list(skill_obj.intents or []),
        "entities": list(skill_obj.entities or []),
        "intent_entity_requirements": dict(getattr(skill_obj, "intent_entity_requirements", {}) or {}),
        "planner": dict(skill_obj.planner_policy or {}),
        "summarizer": dict(skill_obj.summarizer_policy or {}),
        "validator": dict(skill_obj.validator_policy or {}),
        "ui": dict(skill_obj.ui or {}),
        "capability_contract": dict(getattr(skill_obj, "capability_contract", {}) or {}),
        "consistency_report": dict(getattr(skill_obj, "consistency_report", {}) or {}),
    }


def _normalize_entity_groups(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, List[str]] = {}
    for key, items in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(items, list):
            normalized[key_text] = [str(item).strip() for item in items if str(item).strip()]
        elif items not in (None, ""):
            normalized[key_text] = [str(items).strip()]
    return {key: values for key, values in normalized.items() if values}


def _normalize_memory_context(memory_context: Any) -> Dict[str, Any]:
    if not isinstance(memory_context, dict):
        return {}
    working = memory_context.get("working_memory")
    if not isinstance(working, dict):
        return {}
    return {
        "active_entities_by_type": _normalize_entity_groups(working.get("active_entities_by_type")),
        "last_listed_entities_by_type": _normalize_entity_groups(working.get("last_listed_entities_by_type")),
        "recent_entities_by_type": _normalize_entity_groups(working.get("recent_entities_by_type")),
        "last_focus_entity_by_type": {
            str(key).strip(): str(value).strip()
            for key, value in (working.get("last_focus_entity_by_type") or {}).items()
            if str(key).strip() and str(value).strip()
        },
        "last_finance_artifact": dict(working.get("last_finance_artifact") or {})
        if isinstance(working.get("last_finance_artifact"), dict)
        else {},
        "recent_finance_results": list(working.get("recent_finance_results") or []),
        "last_operation": str(working.get("last_operation") or "").strip(),
        "focus_stack": list(working.get("focus_stack") or []),
    }


def _infer_intents_for_subquery(subquery: str, selected_skill: str) -> List[str]:
    text = (subquery or "").lower()
    skill = (selected_skill or "").lower()
    if skill == "semantic_researcher":
        if any(tok in text for tok in ["resumen", "resume", "sintetiza", "conclusion"]):
            return ["semantic_synthesis"]
        return ["semantic_lookup"]
    if skill == "contabilidad_automatica":
        if any(tok in text for tok in ["flujo", "flujos"]) and (
            _looks_like_referential_request(text) or "concili" in text or "loc-" in text
        ):
            return ["explain_reconciliation_flows"]
        if any(tok in text for tok in ["como llegaste", "cómo llegaste", "detalle", "detall", "explicame", "explícame"]) and (
            _looks_like_referential_request(text)
            or "loc-" in text
            or "concili" in text
            or "cuadrado" in text
            or "diferencia de saldo" in text
        ):
            return ["explain_reconciliation_result"]
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


def _normalize_declared_entities(world_contract: Dict[str, Any]) -> List[str]:
    entities = world_contract.get("entities") if isinstance(world_contract.get("entities"), list) else []
    return [str(entity).strip() for entity in entities if str(entity).strip()]


def _looks_like_batch_request(text: str) -> bool:
    lowered = (text or "").lower()
    return any(token in lowered for token in _PLURAL_BATCH_HINTS)


def _contains_hint_token(text: str, token: str) -> bool:
    lowered = (text or "").lower()
    normalized_token = (token or "").strip().lower()
    if not lowered or not normalized_token:
        return False
    pattern = r"(?<!\w)" + re.escape(normalized_token).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, lowered, flags=re.IGNORECASE) is not None


def _looks_like_referential_request(text: str) -> bool:
    return any(_contains_hint_token(text, token) for token in _REFERENTIAL_HINTS) or bool(
        _REFERENTIAL_PRONOUN_RE.search(text or "")
    )


def _looks_like_singular_referential_request(text: str) -> bool:
    return any(_contains_hint_token(text, token) for token in _SINGULAR_REFERENTIAL_HINTS)


def _extract_json_entity_batches(text: str, declared_entities: List[str]) -> Dict[str, List[str]]:
    if not declared_entities:
        return {}
    batches: Dict[str, List[str]] = {}
    for chunk in re.findall(r"\{[^{}]+\}", text or ""):
        try:
            parsed = json.loads(chunk)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        for entity_name in declared_entities:
            if entity_name in parsed and parsed[entity_name] not in (None, ""):
                batches.setdefault(entity_name, []).append(str(parsed[entity_name]).strip())
    return {key: list(dict.fromkeys(values)) for key, values in batches.items() if values}


def _extract_explicit_entity_mentions(text: str, declared_entities: List[str]) -> Dict[str, List[str]]:
    batches: Dict[str, List[str]] = {}
    for entity_name in declared_entities:
        pattern = re.compile(rf"\b{re.escape(entity_name)}\b\s*[:=]\s*([A-Za-z0-9_.-]+)", flags=re.IGNORECASE)
        matches = [match.group(1).strip() for match in pattern.finditer(text or "") if match.group(1).strip()]
        if matches:
            batches[entity_name] = list(dict.fromkeys(matches))
    return batches


def _extract_generic_id_batches(
    text: str,
    declared_entities: List[str],
    existing_entities: Dict[str, Any],
) -> Dict[str, List[str]]:
    id_entities = [entity for entity in declared_entities if entity.endswith("_id")]
    if len(id_entities) != 1:
        return {}
    target_entity = id_entities[0]
    if existing_entities.get(target_entity) not in (None, ""):
        existing_value = str(existing_entities[target_entity]).strip()
    else:
        existing_value = ""
    candidates = [
        match.group(0).strip()
        for match in _GENERIC_ID_TOKEN_RE.finditer(text or "")
        if match.group(0).strip()
    ]
    if existing_value and existing_value not in candidates:
        candidates.append(existing_value)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) <= 1:
        return {}
    return {target_entity: candidates}


def _extract_batch_entity_groups(
    *,
    subquery_text: str,
    world_contract: Dict[str, Any],
    entities: Dict[str, Any],
) -> Dict[str, List[str]]:
    declared_entities = _normalize_declared_entities(world_contract)
    if not declared_entities:
        return {}

    for extractor in (
        _extract_json_entity_batches,
        _extract_explicit_entity_mentions,
    ):
        groups = extractor(subquery_text, declared_entities)
        multi = {key: values for key, values in groups.items() if len(values) > 1}
        if multi:
            return multi

    if _looks_like_batch_request(subquery_text):
        groups = _extract_generic_id_batches(subquery_text, declared_entities, entities)
        multi = {key: values for key, values in groups.items() if len(values) > 1}
        if multi:
            return multi
    return {}


def _infer_batch_intent(current_intents: List[str], world_contract: Dict[str, Any]) -> str:
    world_intents = world_contract.get("intents") if isinstance(world_contract.get("intents"), list) else []
    normalized_world_intents = [str(intent).strip() for intent in world_intents if str(intent).strip()]
    if any(intent.startswith("batch_") for intent in current_intents):
        return current_intents[0]
    if len(normalized_world_intents) == 1 and normalized_world_intents[0].startswith("batch_"):
        return normalized_world_intents[0]
    if "reconcile_credit" in current_intents and "batch_reconcile" in normalized_world_intents:
        return "batch_reconcile"
    if any(intent in {"query_data", "query_financial_data"} for intent in current_intents):
        if "batch_query" in normalized_world_intents:
            return "batch_query"
    batch_intents = [intent for intent in normalized_world_intents if intent.startswith("batch_")]
    return batch_intents[0] if len(batch_intents) == 1 else ""


def _strip_batch_values_from_text(base_text: str, entity_values: List[str]) -> str:
    stripped = base_text or ""
    for value in entity_values:
        if not value:
            continue
        stripped = re.sub(rf"\b{re.escape(value)}\b", "", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    stripped = re.sub(r"\s+([,;:])", r"\1", stripped)
    stripped = re.sub(r"([:;,]){2,}", r"\1", stripped)
    return stripped.strip(" ,;")


def _expand_batch_propositions(
    *,
    world_contract: Dict[str, Any],
    subqueries: List[str],
    subquery_intents: List[List[str]],
    entities_by_subquery: List[Dict[str, Any]],
    required_sources_by_subquery: List[List[str]],
) -> Tuple[List[str], List[List[str]], List[Dict[str, Any]], List[List[str]], str]:
    if len(subqueries) != 1:
        return subqueries, subquery_intents, entities_by_subquery, required_sources_by_subquery, "as_is"
    base_subquery = subqueries[0]
    current_intents = subquery_intents[0] if subquery_intents else []
    current_entities = entities_by_subquery[0] if entities_by_subquery else {}
    batch_entity_groups = _extract_batch_entity_groups(
        subquery_text=base_subquery,
        world_contract=world_contract,
        entities=current_entities,
    )
    if not batch_entity_groups:
        return subqueries, subquery_intents, entities_by_subquery, required_sources_by_subquery, "as_is"

    batch_intent = _infer_batch_intent(current_intents, world_contract)
    if not batch_intent:
        return subqueries, subquery_intents, entities_by_subquery, required_sources_by_subquery, "as_is"

    entity_name, entity_values = next(iter(batch_entity_groups.items()))
    cleaned_base = _strip_batch_values_from_text(base_subquery, entity_values)
    if not cleaned_base:
        cleaned_base = base_subquery

    expanded_subqueries: List[str] = []
    expanded_intents: List[List[str]] = []
    expanded_entities: List[Dict[str, Any]] = []
    expanded_required_sources: List[List[str]] = []
    inherited_sources = required_sources_by_subquery[0] if required_sources_by_subquery else []
    inherited_entities = {
        key: value
        for key, value in current_entities.items()
        if key != entity_name and key != "db_files"
    }
    inherited_db_files = current_entities.get("db_files", []) if isinstance(current_entities.get("db_files"), list) else []

    for entity_value in entity_values:
        expanded_subqueries.append(f'{cleaned_base} {json.dumps({entity_name: entity_value}, ensure_ascii=False)}'.strip())
        expanded_intents.append([current_intents[0]] if current_intents else [batch_intent])
        entity_payload = dict(inherited_entities)
        entity_payload[entity_name] = entity_value
        if inherited_db_files:
            entity_payload["db_files"] = list(inherited_db_files)
        expanded_entities.append(entity_payload)
        expanded_required_sources.append(list(inherited_sources))

    return (
        expanded_subqueries,
        expanded_intents,
        expanded_entities,
        expanded_required_sources,
        "batch_entity_split",
    )


def _resolve_entities_from_memory(
    *,
    subquery_text: str,
    world_contract: Dict[str, Any],
    entities: Dict[str, Any],
    memory_context: Dict[str, Any],
) -> Dict[str, List[str]]:
    declared_entities = _normalize_declared_entities(world_contract)
    if not declared_entities:
        return {}
    if any(entity_name in entities and entities.get(entity_name) not in (None, "") for entity_name in declared_entities):
        return {}
    if not _looks_like_referential_request(subquery_text):
        return {}

    normalized_memory = _normalize_memory_context(memory_context)
    if _looks_like_singular_referential_request(subquery_text):
        last_focus = normalized_memory.get("last_focus_entity_by_type", {})
        if isinstance(last_focus, dict):
            for entity_name in declared_entities:
                value = str(last_focus.get(entity_name) or "").strip()
                if value:
                    return {entity_name: [value]}
        finance_artifact = normalized_memory.get("last_finance_artifact", {})
        if isinstance(finance_artifact, dict):
            for entity_name in declared_entities:
                value = str(finance_artifact.get(entity_name) or "").strip()
                if value:
                    return {entity_name: [value]}
        recent_finance = normalized_memory.get("recent_finance_results", [])
        if isinstance(recent_finance, list):
            for item in reversed(recent_finance):
                if not isinstance(item, dict):
                    continue
                for entity_name in declared_entities:
                    value = str(item.get(entity_name) or "").strip()
                    if value:
                        return {entity_name: [value]}
    sources = [
        normalized_memory.get("last_listed_entities_by_type", {}),
        normalized_memory.get("active_entities_by_type", {}),
        normalized_memory.get("recent_entities_by_type", {}),
    ]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for entity_name in declared_entities:
            values = source.get(entity_name)
            if isinstance(values, list) and values:
                return {entity_name: [str(value).strip() for value in values if str(value).strip()]}

    focus_stack = normalized_memory.get("focus_stack", [])
    if isinstance(focus_stack, list):
        for focus in reversed(focus_stack):
            if not isinstance(focus, dict):
                continue
            focus_entities = _normalize_entity_groups(focus.get("entities"))
            for entity_name in declared_entities:
                values = focus_entities.get(entity_name)
                if values:
                    return {entity_name: values}
    return {}


def _expand_memory_references(
    *,
    subqueries: List[str],
    subquery_intents: List[List[str]],
    entities_by_subquery: List[Dict[str, Any]],
    required_sources_by_subquery: List[List[str]],
    world_contract: Dict[str, Any],
    memory_context: Dict[str, Any],
) -> Tuple[List[str], List[List[str]], List[Dict[str, Any]], List[List[str]], str]:
    expanded_subqueries: List[str] = []
    expanded_intents: List[List[str]] = []
    expanded_entities: List[Dict[str, Any]] = []
    expanded_sources: List[List[str]] = []
    changed = False

    for subquery, intents, entities, required_sources in zip(
        subqueries,
        subquery_intents,
        entities_by_subquery,
        required_sources_by_subquery,
    ):
        memory_entities = _resolve_entities_from_memory(
            subquery_text=subquery,
            world_contract=world_contract,
            entities=entities,
            memory_context=memory_context,
        )
        if not memory_entities:
            expanded_subqueries.append(subquery)
            expanded_intents.append(intents)
            expanded_entities.append(entities)
            expanded_sources.append(required_sources)
            continue

        entity_name, entity_values = next(iter(memory_entities.items()))
        if len(entity_values) == 1:
            merged_entities = dict(entities)
            merged_entities[entity_name] = entity_values[0]
            expanded_subqueries.append(f'{subquery} {json.dumps({entity_name: entity_values[0]}, ensure_ascii=False)}'.strip())
            expanded_intents.append(intents)
            expanded_entities.append(merged_entities)
            expanded_sources.append(required_sources)
            changed = True
            continue

        batch_intent = _infer_batch_intent(intents, world_contract)
        if not batch_intent:
            expanded_subqueries.append(subquery)
            expanded_intents.append(intents)
            expanded_entities.append(entities)
            expanded_sources.append(required_sources)
            continue

        changed = True
        for entity_value in entity_values:
            merged_entities = dict(entities)
            merged_entities[entity_name] = entity_value
            expanded_subqueries.append(f'{subquery} {json.dumps({entity_name: entity_value}, ensure_ascii=False)}'.strip())
            expanded_intents.append([intents[0]] if intents else [batch_intent])
            expanded_entities.append(merged_entities)
            expanded_sources.append(list(required_sources))

    return (
        expanded_subqueries,
        expanded_intents,
        expanded_entities,
        expanded_sources,
        "memory_reference_resolution" if changed else "as_is",
    )


def _infer_finance_required_sources(subquery: str, intents: List[str]) -> List[str]:
    text = (subquery or "").lower()
    if any(
        intent in {
            "reconcile_credit",
            "audit_drift",
            "batch_reconcile",
            "explain_reconciliation_result",
            "explain_reconciliation_flows",
        }
        for intent in intents
    ):
        return ["contabilidad.db", "transacciones.db"]
    explicit_db_matches = re.findall(r"\b([A-Za-z0-9_.-]+\.db)\b", subquery or "", flags=re.IGNORECASE)
    if explicit_db_matches:
        return [db.lower() for db in explicit_db_matches]
    if _contains_any(
        text,
        [
            "bases de datos",
            "base de datos",
            "tus bases",
            "todas las fuentes",
            "todas las bases",
            "de tus datos",
            "informacion completa",
            "información completa",
            "perfil completo",
            "overview completo",
        ],
    ):
        return ["contabilidad.db", "transacciones.db"]
    if _contains_any(text, ["movimiento", "movimientos", "transaccion", "transacciones", "desembolso", "pago"]):
        return ["transacciones.db"]
    return ["contabilidad.db"]


def _derive_source_scope(required_sources_by_subquery: List[List[str]]) -> str:
    flattened = {
        source.lower()
        for group in required_sources_by_subquery
        for source in group
        if isinstance(source, str) and source.strip()
    }
    return "multi_source" if len(flattened) > 1 else "single_source"


def _derive_composition_mode(
    *,
    selected_skill_world: str,
    subquery_intents: List[List[str]],
    required_sources_by_subquery: List[List[str]],
) -> str:
    flat_intents = {intent for intents in subquery_intents for intent in intents}
    flat_sources = {
        source.lower()
        for group in required_sources_by_subquery
        for source in group
        if isinstance(source, str)
    }
    if any(
        intent in {
            "reconcile_credit",
            "audit_drift",
            "batch_reconcile",
            "explain_reconciliation_result",
            "explain_reconciliation_flows",
        }
        for intent in flat_intents
    ):
        return "reconcile"
    if selected_skill_world == "contabilidad_automatica" and len(flat_sources) > 1:
        return "merge"
    if any(intent in {"compare_entities"} for intent in flat_intents):
        return "compare"
    return "lookup"


def _derive_coverage_expectation(
    *,
    response_mode: str,
    source_scope: str,
    composition_mode: str,
    subquery_count: int,
) -> str:
    if response_mode == "audit":
        return "exhaustive"
    if composition_mode in {"reconcile", "merge"} or source_scope == "multi_source" or subquery_count > 1:
        return "composite"
    return "summary"


def _derive_decomposition_strategy(
    *,
    finance_strategy: str,
    memory_strategy: str,
    batch_strategy: str,
    subquery_count: int,
) -> str:
    if batch_strategy != "as_is":
        return batch_strategy
    if memory_strategy != "as_is":
        if subquery_count > 1:
            return "memory_reference_batch_split"
        return memory_strategy
    return finance_strategy


def _expand_finance_propositions(
    *,
    user_prompt: str,
    selected_skill_world: str,
    subqueries: List[str],
    subquery_intents: List[List[str]],
    entities_by_subquery: List[Dict[str, Any]],
    required_sources_by_subquery: List[List[str]],
) -> Tuple[List[str], List[List[str]], List[Dict[str, Any]], List[List[str]], str]:
    if selected_skill_world != "contabilidad_automatica":
        return subqueries, subquery_intents, entities_by_subquery, required_sources_by_subquery, "as_is"
    if len(subqueries) != 1:
        return subqueries, subquery_intents, entities_by_subquery, required_sources_by_subquery, "as_is"

    intents = subquery_intents[0] if subquery_intents else []
    entities = entities_by_subquery[0] if entities_by_subquery else {}
    required_sources = required_sources_by_subquery[0] if required_sources_by_subquery else []
    credito_id = str(entities.get("credito_id") or "").strip()

    if not credito_id or intents != ["query_financial_data"]:
        return subqueries, subquery_intents, entities_by_subquery, required_sources_by_subquery, "as_is"
    if sorted({source.lower() for source in required_sources}) != ["contabilidad.db", "transacciones.db"]:
        return subqueries, subquery_intents, entities_by_subquery, required_sources_by_subquery, "as_is"

    expanded_subqueries = [
        f"snapshot contable del crédito {credito_id} en contabilidad.db",
        f"movimientos del crédito {credito_id} en transacciones.db",
    ]
    expanded_intents = [["query_financial_data"], ["query_financial_data"]]
    expanded_entities = [
        {"credito_id": credito_id, "db_files": ["contabilidad.db"]},
        {"credito_id": credito_id, "db_files": ["transacciones.db"]},
    ]
    expanded_required_sources = [["contabilidad.db"], ["transacciones.db"]]

    if _contains_any(user_prompt, ["pago total", "pagado", "cuanto se ha pagado", "cuánto se ha pagado"]):
        expanded_subqueries[1] = f"pagos del crédito {credito_id} en transacciones.db"

    return expanded_subqueries, expanded_intents, expanded_entities, expanded_required_sources, "finance_cross_source_split"


def _extract_entities(subquery: str, declared_entities: Optional[List[str]] = None) -> Dict[str, Any]:
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
    normalized_declared = [entity for entity in (declared_entities or []) if entity]
    explicit_mentions = _extract_explicit_entity_mentions(text, normalized_declared)
    for entity_name, values in explicit_mentions.items():
        if values and entity_name not in entities:
            entities[entity_name] = values[0]
    id_entities = [entity for entity in normalized_declared if entity.endswith("_id")]
    if len(id_entities) == 1 and id_entities[0] not in entities:
        generic_tokens = [
            match.group(0).strip()
            for match in _GENERIC_ID_TOKEN_RE.finditer(text)
            if match.group(0).strip()
        ]
        if generic_tokens:
            entities[id_entities[0]] = generic_tokens[0]
    db_matches = re.findall(r"\b([A-Za-z0-9_.-]+\.db)\b", text, flags=re.IGNORECASE)
    if db_matches:
        entities["db_files"] = db_matches
    return entities


def _derive_constraints(
    subquery: str,
    entities: Dict[str, Any],
    intents: List[str],
    required_sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    text = (subquery or "").lower()
    constraints: Dict[str, Any] = {}
    if "batch_query" in intents:
        constraints["batch"] = True
    if "credito_id" in entities:
        constraints["entity_scope"] = "single_credit"
    if required_sources:
        constraints["required_sources"] = list(required_sources)
        constraints["source_scope"] = "multi_source" if len(set(required_sources)) > 1 else "single_source"
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
    memory_context = state.get("memory_context") or {}
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
    if isinstance(memory_context, dict) and memory_context:
        sys_content += (
            "\n\nMEMORY_CONTEXT (ESTRUCTURADO):\n"
            f"{json.dumps(memory_context, ensure_ascii=False, indent=2)}"
        )
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
    active_skill = selected_skills[0] if selected_skills else selected_skill_world
    world_contract = _build_world_contract(skill_registry, active_skill)
    declared_entities = _normalize_declared_entities(world_contract)

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
    subquery_intents = [_infer_intents_for_subquery(sq, active_skill) for sq in subqueries]
    entities_by_subquery = [_extract_entities(sq, declared_entities) for sq in subqueries]
    required_sources_by_subquery: List[List[str]] = []
    for sq, intents, entities in zip(subqueries, subquery_intents, entities_by_subquery):
        if active_skill == "contabilidad_automatica":
            required_sources_by_subquery.append(_infer_finance_required_sources(sq, intents))
        else:
            db_files = entities.get("db_files", [])
            required_sources_by_subquery.append(list(db_files) if isinstance(db_files, list) else [])

    (
        subqueries,
        subquery_intents,
        entities_by_subquery,
        required_sources_by_subquery,
        decomposition_strategy,
    ) = _expand_finance_propositions(
        user_prompt=user_prompt,
        selected_skill_world=selected_skill_world,
        subqueries=subqueries,
        subquery_intents=subquery_intents,
        entities_by_subquery=entities_by_subquery,
        required_sources_by_subquery=required_sources_by_subquery,
    )
    (
        subqueries,
        subquery_intents,
        entities_by_subquery,
        required_sources_by_subquery,
        memory_decomposition_strategy,
    ) = _expand_memory_references(
        subqueries=subqueries,
        subquery_intents=subquery_intents,
        entities_by_subquery=entities_by_subquery,
        required_sources_by_subquery=required_sources_by_subquery,
        world_contract=world_contract,
        memory_context=memory_context,
    )
    (
        subqueries,
        subquery_intents,
        entities_by_subquery,
        required_sources_by_subquery,
        batch_decomposition_strategy,
    ) = _expand_batch_propositions(
        world_contract=world_contract,
        subqueries=subqueries,
        subquery_intents=subquery_intents,
        entities_by_subquery=entities_by_subquery,
        required_sources_by_subquery=required_sources_by_subquery,
    )
    subqueries_logic = [f"q{i+1}" for i in range(len(subqueries))]
    logic_form = " AND ".join(subqueries_logic)
    constraints_by_subquery = [
        _derive_constraints(sq, entities, intents, required_sources)
        for sq, entities, intents, required_sources in zip(
            subqueries,
            entities_by_subquery,
            subquery_intents,
            required_sources_by_subquery,
        )
    ]
    response_mode = _guess_response_mode(user_prompt, subqueries)
    source_scope = _derive_source_scope(required_sources_by_subquery)
    composition_mode = _derive_composition_mode(
        selected_skill_world=selected_skill_world,
        subquery_intents=subquery_intents,
        required_sources_by_subquery=required_sources_by_subquery,
    )
    coverage_expectation = _derive_coverage_expectation(
        response_mode=response_mode,
        source_scope=source_scope,
        composition_mode=composition_mode,
        subquery_count=len(subqueries),
    )
    selection_mode = "forced" if normalized_allowlist else "auto"
    analyzer = {
        "input_payload": {"user_prompt": user_prompt},
        "selected_skill_world": selected_skill_world,
        "selection_mode": selection_mode,
        "propositional_logic": logic_form,
        "subqueries": subqueries,
        "subqueries_logic": subqueries_logic,
        "subquery_intents": subquery_intents,
        "entities_by_subquery": entities_by_subquery,
        "required_sources_by_subquery": required_sources_by_subquery,
        "constraints_by_subquery": constraints_by_subquery,
        "source_scope": source_scope,
        "composition_mode": composition_mode,
        "coverage_expectation": coverage_expectation,
        "decomposition_strategy": _derive_decomposition_strategy(
            finance_strategy=decomposition_strategy,
            memory_strategy=memory_decomposition_strategy,
            batch_strategy=batch_decomposition_strategy,
            subquery_count=len(subqueries),
        ),
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
        "required_sources_by_subquery": required_sources_by_subquery,
        "constraints_by_subquery": constraints_by_subquery,
        "world_contract": world_contract,
        "messages": [analyzer_msg],
    }

