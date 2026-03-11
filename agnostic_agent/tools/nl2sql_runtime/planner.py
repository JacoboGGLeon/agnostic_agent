from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StepNode:
    id: str
    intent: str
    retrieval_query: str
    k: int = 1
    depends_on: List[str] = field(default_factory=list)


@dataclass
class GraphPlan:
    nodes: List[StepNode]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "intent": n.intent,
                    "retrieval_query": n.retrieval_query,
                    "k": n.k,
                    "depends_on": list(n.depends_on),
                }
                for n in self.nodes
            ]
        }


class RouterPlanner:
    def __init__(self, *, k_min: int = 1, k_max: int = 5) -> None:
        self.k_min = max(1, int(k_min))
        self.k_max = max(self.k_min, int(k_max))

    def plan(self, user_query: str, k: int = 5) -> GraphPlan:
        query = user_query or ""
        low = query.lower()
        k_eff = max(self.k_min, min(int(k or self.k_min), self.k_max))
        intents: List[str] = ["lookup_tables", "lookup_columns"]
        join_triggers = [
            "join",
            "combina",
            "combinar",
            "relaciona",
            "relacion",
            "cruza",
            "contra",
            "versus",
            " vs ",
            "junto con",
        ]
        if any(tok in low for tok in join_triggers):
            intents.append("join")
        operation_triggers = ["count", "cuantos", "cuántos", "sum", "suma", "avg", "promedio", "group by", "agrupa", "por "]
        if any(tok in low for tok in operation_triggers):
            intents.append("operation")
        elif "operation" not in intents:
            intents.append("operation")
        nodes: List[StepNode] = []
        prev = ""
        for idx, intent in enumerate(intents, start=1):
            node = StepNode(
                id=f"step{idx}",
                intent=intent,
                retrieval_query=query,
                k=k_eff if intent.startswith("lookup") else self.k_min,
                depends_on=[prev] if prev else [],
            )
            nodes.append(node)
            prev = node.id
        return GraphPlan(nodes=nodes)
