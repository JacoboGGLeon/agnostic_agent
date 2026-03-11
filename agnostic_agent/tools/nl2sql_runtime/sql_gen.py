from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from agnostic_agent.providers.factory import ProviderFactory


def _one_line_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", str(sql or "").strip()).strip()


def _heuristic_fallback(user_query: str, outputs: Dict[str, Any], row_limit: int, entity_id: str = "") -> Dict[str, Any]:
    primary_table = None
    primary_columns: List[Dict[str, Any]] = []
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
    if any(tok in req for tok in ["count", "cuantos", "cuántos"]):
        select_expr = "COUNT(*) AS total_rows"
    elif any(tok in req for tok in ["sum", "suma", "total", "promedio", "avg", "average"]):
        numeric = next((c for c in columns if any(tok in c.lower() for tok in ["monto", "saldo", "total", "capital", "interes"])), None)
        if any(tok in req for tok in ["promedio", "avg", "average"]) and numeric:
            select_expr = f"AVG({numeric}) AS avg_{numeric}"
        elif numeric:
            select_expr = f"SUM({numeric}) AS sum_{numeric}"
        else:
            select_expr = ", ".join(columns[:6]) if columns else "*"
    else:
        select_expr = ", ".join(columns[:6]) if columns else "*"
    where_bits: List[str] = []
    if entity_id:
        target_col = next((c for c in columns if "credito" in c.lower() and "id" in c.lower()), "")
        if not target_col:
            target_col = "credito_id"
        if target_col:
            where_bits.append(f"{target_col} = '{entity_id}'")
    sql = f"SELECT {select_expr} FROM {table_name}"
    if where_bits:
        sql += " WHERE " + " AND ".join(where_bits)
    sql += f" LIMIT {max(1, min(int(row_limit or 50), 500))}"
    return {
        "sql_supposed": "heuristic_fallback",
        "sql_proposal": _one_line_sql(sql),
        "where_clauses": where_bits,
        "chosen_table": table_name,
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
