from agnostic_agent.protocols.tcp import ToolContract, ToolNormalizedResult


def test_tcp_contract_defaults():
    contract = ToolContract(
        name="query_transactions_db",
        description="Query transaction movements",
        input_schema="schemas/query.input.json",
        output_schema="schemas/query.output.json",
    )
    assert contract.side_effects == "read_only"
    assert contract.timeout_s == 30.0
    assert contract.testing.mode == "explicit_or_auto"


def test_tcp_normalized_result_contract():
    result = ToolNormalizedResult(ok=False, error={"code": "timeout", "message": "slow"})
    assert result.ok is False
    assert result.error is not None
    assert result.error["code"] == "timeout"
