from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from agnostic_agent.providers.factory import ProviderFactory


def _one_line_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", str(sql or "").strip()).strip()


def _first_payload(outputs: Dict[str, Any], payload_type: str) -> Dict[str, Any]:
    for payload in outputs.values():
        if isinstance(payload, dict) and payload.get("type") == payload_type:
            return payload
    return {}


def _choose_group_column(query: str, columns: List[str]) -> str:
    q = (query or "").lower()
    for token in ["estatus", "tipo", "fecha", "cliente", "credito"]:
        if token in q:
            hit = next((c for c in columns if token in c.lower()), "")
            if hit:
                return hit
    if any(tok in q for tok in ["group by", "agrupa", " por "]):
        return next(
            (
                c
                for c in columns
                if not any(num in c.lower() for num in ["monto", "saldo", "total", "capital", "interes"])
            ),
            "",
        )
    return ""


def _heuristic_fallback(user_query: str, outputs: Dict[str, Any], row_limit: int, entity_id: str = "") -> Dict[str, Any]:
    primary_table = None
    primary_columns: List[Dict[str, Any]] = []
    relationship_payload = _first_payload(outputs, "relationship_payload")
    relationship_primary = relationship_payload.get("primary") if isinstance(relationship_payload, dict) else None
    for payload in outputs.values():
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "table_payload" and isinstance(payload.get("primary"), dict) and primary_table is None:
            primary_table = payload["primary"]
        if payload.get("type") in {"columns_payload", "operation_payload"}:
            candidates = payload.get("candidates") or []
            if isinstance(candidates, list):
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        primary_columns.append(candidate)

    table_name = str((primary_table or {}).get("table") or "sqlite_master")
    columns = [str(c.get("column") or c.get("name") or "") for c in primary_columns if str(c.get("column") or c.get("name") or "")]
    columns = [c for idx, c in enumerate(columns) if c and c not in columns[:idx]]
    req = (user_query or "").lower()
    group_col = _choose_group_column(req, columns)
    numeric = next((c for c in columns if any(tok in c.lower() for tok in ["monto", "saldo", "total", "capital", "interes"])), None)

    if any(tok in req for tok in ["count", "cuantos", "cuántos"]):
        select_expr = f"{group_col}, COUNT(*) AS total_rows" if group_col else "COUNT(*) AS total_rows"
    elif any(tok in req for tok in ["sum", "suma", "total", "promedio", "avg", "average"]):
        if any(tok in req for tok in ["promedio", "avg", "average"]) and numeric:
            agg_expr = f"AVG({numeric}) AS avg_{numeric}"
        elif numeric:
            agg_expr = f"SUM({numeric}) AS sum_{numeric}"
        else:
            agg_expr = ", ".join(columns[:6]) if columns else "*"
        select_expr = f"{group_col}, {agg_expr}" if group_col and agg_expr else agg_expr
    else:
        select_expr = ", ".join(columns[:6]) if columns else "*"

    from_expr = table_name
    chosen_table = table_name
    relationship_meta = relationship_primary.get("rich_context") if isinstance(relationship_primary, dict) and isinstance(relationship_primary.get("rich_context"), dict) else {}
    if relationship_meta:
        left_table = str(relationship_meta.get("left_table") or table_name)
        left_col = str(relationship_meta.get("left_column") or "")
        right_table = str(relationship_meta.get("right_table") or "")
        right_col = str(relationship_meta.get("right_column") or "")
        if left_table and left_col and right_table and right_col:
            from_expr = f"{left_table} t1 JOIN {right_table} t2 ON t1.{left_col} = t2.{right_col}"
            chosen_table = f"{left_table}_join_{right_table}"

    where_bits: List[str] = []
    if entity_id:
        target_col = next((c for c in columns if "credito" in c.lower() and "id" in c.lower()), "")
        if not target_col:
            target_col = "credito_id"
        if target_col:
            where_bits.append(f"{'t1.' if relationship_meta else ''}{target_col} = '{entity_id}'")

    sql = f"SELECT {select_expr} FROM {from_expr}"
    if where_bits:
        sql += " WHERE " + " AND ".join(where_bits)
    if group_col and any(tok in req for tok in ["count", "cuantos", "cuántos", "sum", "suma", "promedio", "avg", "average", "group by", "agrupa", " por "]):
        sql += f" GROUP BY {group_col}"
    if any(tok in req for tok in ["ultimo", "último", "reciente", "latest"]) and any(c.lower() == "fecha" for c in columns):
        sql += " ORDER BY fecha DESC"
    sql += f" LIMIT {max(1, min(int(row_limit or 50), 500))}"
    return {
        "sql_supposed": "heuristic_fallback",
        "sql_proposal": _one_line_sql(sql),
        "where_clauses": where_bits,
        "chosen_table": chosen_table,
    }


def _provider_cfg() -> Dict[str, Any]:
    provider = os.getenv("KNOWLEDGE_VOYAGUE_NL2SQL_LLM_PROVIDER", "").strip().lower()
    if not provider:
        if os.getenv("BEDROCK_MODEL_ID", "").strip():
            provider = "bedrock"
        elif os.getenv("VLLM_API_BASE", "").strip():
            provider = "vllm"
        else:
            provider = "openai"
    if provider == "bedrock":
        return {
            "provider": "bedrock",
            "model": os.getenv("KNOWLEDGE_VOYAGUE_BEDROCK_MODEL", "").strip() or os.getenv("BEDROCK_MODEL_ID", "").strip() or "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "region_name": os.getenv("BEDROCK_REGION", "").strip() or os.getenv("AWS_REGION", "").strip(),
            "api": os.getenv("BEDROCK_API_MODE", "").strip() or "auto",
        }
    if provider == "vllm":
        return {
            "provider": "vllm",
            "model": os.getenv("KNOWLEDGE_VOYAGUE_NL2SQL_MODEL", "").strip() or os.getenv("LLM_SERVED_NAME", "").strip() or "agnostic-llm",
            "base_url": os.getenv("VLLM_API_BASE", "").strip() or "http://127.0.0.1:8000/v1",
            "api_key": os.getenv("OPENAI_API_KEY", "").strip() or "EMPTY",
        }
    return {
        "provider": "openai",
        "model": os.getenv("KNOWLEDGE_VOYAGUE_NL2SQL_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
    }


def generate_sql(*, user_query: str, plan: Dict[str, Any], outputs: Dict[str, Any], row_limit: int, entity_id: str = "") -> Dict[str, Any]:
    compact_outputs = outputs
    system_prompt = (
        "You are an expert SQLite NL2SQL planner.\n"
        "Return ONLY strict JSON with keys sql_supposed and sql_proposal.\n"
        "Rules:\n"
        "- Use the provided plan and payloads.\n"
        "- Generate exactly one read-only SELECT or WITH query.\n"
        f"- Always include LIMIT <= {max(1, min(int(row_limit or 50), 500))}.\n"
        "- Do not include markdown fences.\n"
    )
    prompt = json.dumps(
        {
            "user_query": user_query,
            "plan": plan,
            "retrieval_outputs": compact_outputs,
            "entity_id": entity_id or None,
        },
        ensure_ascii=False,
    )
    try:
        provider = ProviderFactory.get_llm_provider(_provider_cfg())
        raw = provider.generate(prompt, system_prompt=system_prompt, temperature=0.0)
        parsed = json.loads(raw)
        sql = _one_line_sql(str(parsed.get("sql_proposal", "")))
        if not re.match(r"^(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
            raise ValueError("generated SQL is not read-only")
        return {
            "sql_supposed": str(parsed.get("sql_supposed", "")),
            "sql_proposal": sql,
            "provider_raw": raw,
            "where_clauses": [],
            "chosen_table": None,
        }
    except Exception:
        return _heuristic_fallback(user_query, outputs, row_limit, entity_id=entity_id)
