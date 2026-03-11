from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .builder import PayloadBuilder
from .catalog import load_catalog
from .planner import RouterPlanner
from .retriever import SemanticRetriever
from .sql_exec import execute_sql, summarize_result
from .sql_gen import generate_sql


@dataclass
class NL2SQLRuntimeConfig:
    catalog_path: str
    db_path: str
    row_limit: int = 50
    k: int = 5


class NL2SQLRuntimeAgent:
    def __init__(self, cfg: NL2SQLRuntimeConfig) -> None:
        self.cfg = cfg
        self.catalog = load_catalog(cfg.catalog_path)
        self.planner = RouterPlanner(k_min=1, k_max=max(1, min(cfg.k, 8)))
        self.retriever = SemanticRetriever(self.catalog)
        self.builder = PayloadBuilder(self.catalog)

    def query(self, *, user_query: str, execute: bool = False, entity_id: str = "") -> Dict[str, Any]:
        plan = self.planner.plan(user_query, k=self.cfg.k)
        plan_dict = plan.to_dict()
        ctx_by_node: Dict[str, Dict[str, Any]] = {}
        for node in plan_dict.get("nodes", []):
            node_id = node.get("id")
            if not node_id:
                continue
            ctx_by_node[node_id] = self.retriever.get_context_rich(
                intent=str(node.get("intent", "")),
                retrieval_query=str(node.get("retrieval_query", "")),
                k=int(node.get("k", self.cfg.k)),
            )
        outputs = self.builder.build_for_graph(nodes=plan_dict.get("nodes", []), ctx_by_node=ctx_by_node, top_k=self.cfg.k).get("nodes_payload", {})
        gen = generate_sql(
            user_query=user_query,
            plan=plan_dict,
            outputs=outputs,
            row_limit=self.cfg.row_limit,
            entity_id=entity_id,
        )
        result: Dict[str, Any] = {
            "ok": True,
            "agent": "nl2sql_runtime_v4",
            "db_path": self.cfg.db_path,
            "catalog_path": self.cfg.catalog_path,
            "catalog": self.catalog,
            "user_request": user_query,
            "entity_id": entity_id or None,
            "plan": plan_dict,
            "output": outputs,
            "sql_supposed": gen.get("sql_supposed", ""),
            "generated_sql": gen.get("sql_proposal", ""),
            "where_clauses": list(gen.get("where_clauses") or []),
        }
        primary_table = None
        for payload in outputs.values():
            if isinstance(payload, dict) and payload.get("type") == "table_payload" and isinstance(payload.get("primary"), dict):
                primary_table = payload["primary"].get("table")
                break
        result["chosen_table"] = gen.get("chosen_table") or primary_table
        if execute:
            exec_out = execute_sql(self.cfg.db_path, str(gen.get("sql_proposal", "")))
            result["execution"] = exec_out
            result["nl_result"] = summarize_result(user_query, exec_out)
        return result
