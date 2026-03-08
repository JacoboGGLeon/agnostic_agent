from agnostic_agent.protocols.kap import (
    KnowledgeAdapterContract,
    validate_kap_adapter_instance,
    validate_knowledge_item_payload,
)


class _GoodAdapter:
    def search(self, query, **kwargs):
        return []

    def get(self, identifier):
        return {
            "id": identifier,
            "type": "markdown_chunk",
            "content": "x",
            "source": "FinanceRulesKB",
            "metadata": {},
            "provenance": {},
        }


class _BadAdapter:
    pass


def test_kap_contract_defaults():
    contract = KnowledgeAdapterContract(
        name="finance_kb",
        description="Knowledge adapter for finance docs",
        entrypoint="adapters.finance:build",
    )
    assert contract.testing.mode == "explicit_or_auto"


def test_kap_adapter_instance_validation():
    ok, errors = validate_kap_adapter_instance(_GoodAdapter())
    assert ok is True
    assert errors == []

    ok, errors = validate_kap_adapter_instance(_BadAdapter())
    assert ok is False
    assert len(errors) == 2


def test_kap_knowledge_item_payload_validation():
    ok, errors = validate_knowledge_item_payload(
        {
            "id": "chunk_1",
            "type": "markdown_chunk",
            "content": "hola",
            "source": "FinanceRulesKB",
            "metadata": {"lang": "es"},
            "provenance": {"source_path": "rules.md", "chunk_id": "chunk_1"},
        }
    )
    assert ok is True
    assert errors == []
