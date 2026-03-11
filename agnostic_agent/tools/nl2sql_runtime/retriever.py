from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .catalog import catalog_items


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_áéíóúñ]+", (text or "").lower())


def _score(query: str, item: Dict[str, Any]) -> float:
    q = set(_tokenize(query))
    rich = item.get("rich_context") if isinstance(item.get("rich_context"), dict) else {}
    ctx = " ".join(
        [
            str(item.get("table", "")),
            str(item.get("column", "")),
            str(item.get("description", "")),
            str(item.get("planner_context", "")),
            str(rich.get("left_table", "")),
            str(rich.get("left_column", "")),
            str(rich.get("right_table", "")),
            str(rich.get("right_column", "")),
            " ".join(f"{k} {v}" for k, v in (item.get("business_glossary") or {}).items()),
        ]
    )
    tokens = set(_tokenize(ctx))
    overlap = len(q.intersection(tokens))
    if not q:
        return 0.0
    score = overlap / math.sqrt(max(1, len(q)) * max(1, len(tokens)))
    if item.get("type") == "join" and rich:
        left_tokens = set(_tokenize(f"{rich.get('left_table', '')} {rich.get('left_column', '')}"))
        right_tokens = set(_tokenize(f"{rich.get('right_table', '')} {rich.get('right_column', '')}"))
        if q.intersection(left_tokens) and q.intersection(right_tokens):
            score += 0.25
        if any(tok in q for tok in {"join", "combina", "combinar", "relaciona", "cruza", "vs", "contra"}):
            score += 0.15
    return score


@dataclass
class SemanticRetriever:
    catalog: Dict[str, Any]

    def __post_init__(self) -> None:
        self.items = catalog_items(self.catalog)

    def _best(self, item_type: str, retrieval_query: str, k: int) -> List[Dict[str, Any]]:
        candidates = [it for it in self.items if it.get("type") == item_type]
        scored = sorted(
            (
                {
                    **it,
                    "score": _score(retrieval_query, it),
                }
                for it in candidates
            ),
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        return scored[: max(1, int(k or 1))]

    def get_context_rich(self, *, intent: str, retrieval_query: str, k: int) -> Dict[str, Any]:
        if intent == "lookup_tables":
            tables = self._best("table", retrieval_query, k)
            return {"tables": tables, "context": tables}
        if intent == "lookup_columns":
            columns = self._best("column", retrieval_query, k)
            return {"columns": columns, "context": columns}
        if intent == "join":
            joins = self._best("join", retrieval_query, k)
            tables = self._best("table", retrieval_query, k)
            return {"joins": joins, "tables": tables, "context": joins}
        columns = self._best("column", retrieval_query, k)
        tables = self._best("table", retrieval_query, 1)
        return {"columns": columns, "tables": tables, "context": columns}
