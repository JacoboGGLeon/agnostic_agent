from dataclasses import dataclass
from typing import Any, Dict, List


def _infer_intent(request: Dict[str, Any]) -> str:
    explicit = str(request.get("intent") or "").strip()
    if explicit in {"semantic_lookup", "semantic_synthesis"}:
        return explicit
    query = str(request.get("user_request") or request.get("query") or "").lower()
    if any(token in query for token in ["resume", "sintetiza", "sintesis"]):
        return "semantic_synthesis"
    return "semantic_lookup"

@dataclass
class SkillImpl:
    name: str = "semantic_researcher"
    version: str = "1.0.0"

    def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        intent = _infer_intent(request)
        tools: List[str] = ['list_knowledge_sources', 'search_knowledge_base', 'rerank_docs']
        planned = [{'tool': t, 'args': {}} for t in tools]
        return {
            'status': 'success',
            'outputs': {
                'ok': True,
                'intent': intent,
                'summary': f'Skill {self.name} preparada para ejecutar {len(planned)} tool(s).',
                'planned_tool_calls': planned,
                'request': request,
            },
            'artifacts': [],
            'errors': [],
            'metrics': {'planned_calls': len(planned)},
            'children': [],
        }

def build() -> SkillImpl:
    return SkillImpl()
