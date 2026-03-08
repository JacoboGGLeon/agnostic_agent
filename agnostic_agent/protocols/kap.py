from __future__ import annotations

from typing import Any, Dict, Literal, Protocol, Tuple

from pydantic import BaseModel, Field

from agnostic_agent.core.models.runtime_objects import KnowledgeItem


TestMode = Literal["explicit", "auto", "explicit_or_auto"]


class KnowledgeAdapterTesting(BaseModel):
    mode: TestMode = "explicit_or_auto"


class KnowledgeAdapterContract(BaseModel):
    name: str
    description: str
    entrypoint: str
    testing: KnowledgeAdapterTesting = Field(default_factory=KnowledgeAdapterTesting)


class KnowledgeAdapterProtocol(Protocol):
    def search(self, query: str, **kwargs: Any) -> list[KnowledgeItem]: ...

    def get(self, identifier: str) -> KnowledgeItem: ...


class KnowledgeSearchResult(BaseModel):
    items: list[KnowledgeItem] = Field(default_factory=list)


class KnowledgeGetResult(BaseModel):
    item: KnowledgeItem


def validate_kap_adapter_instance(adapter: Any) -> Tuple[bool, list[str]]:
    errors: list[str] = []
    search = getattr(adapter, "search", None)
    get = getattr(adapter, "get", None)
    if not callable(search):
        errors.append("adapter missing callable search(query, **kwargs)")
    if not callable(get):
        errors.append("adapter missing callable get(identifier)")
    return len(errors) == 0, errors


def validate_knowledge_item_payload(payload: Dict[str, Any]) -> Tuple[bool, list[str]]:
    try:
        KnowledgeItem(**payload)
        return True, []
    except Exception as e:
        # Keep deterministic single error line for validator usage.
        return False, [str(e)]
