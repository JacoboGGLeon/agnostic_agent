from __future__ import annotations

from typing import Any, Dict, List


class PayloadBuilder:
    def __init__(self, catalog: Dict[str, Any]) -> None:
        self.catalog = catalog or {}

    def build(self, intent: str, rich_ctx: Dict[str, Any], top_k: int = 1) -> Dict[str, Any]:
        intent = str(intent or "").lower()
        if intent == "lookup_tables":
            tables = list(rich_ctx.get("tables", []) or [])[: max(1, top_k)]
            return {"type": "table_payload", "primary": tables[0] if tables else None, "candidates": tables}
        if intent == "lookup_columns":
            cols = list(rich_ctx.get("columns", []) or [])[: max(1, top_k)]
            return {"type": "columns_payload", "primary": cols[0] if cols else None, "candidates": cols}
        if intent == "join":
            joins = list(rich_ctx.get("joins", []) or [])[: max(1, top_k)]
            return {"type": "relationship_payload", "primary": joins[0] if joins else None, "candidates": joins}
        cols = list(rich_ctx.get("columns", []) or [])[: max(1, top_k)]
        primary = cols[0] if cols else None
        return {"type": "operation_payload", "primary": primary, "candidates": cols}

    def build_for_graph(self, *, nodes: List[Dict[str, Any]], ctx_by_node: Dict[str, Dict[str, Any]], top_k: int = 1) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                continue
            out[node_id] = self.build(
                str(node.get("intent", "")),
                ctx_by_node.get(node_id, {}),
                top_k=int(node.get("k", top_k)),
            )
        return {"nodes_payload": out}
