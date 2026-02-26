# ADR 0003 - Tool Calling Contract

## Status
Accepted

## Context
Providers may emit tool instructions in different formats:
- Native tool calls
- `additional_kwargs.tool_calls`
- XML-like calls
- JSON-in-text (`tool_uses`) fallback

Without a strict contract, execution and UI diverge.

## Decision
Canonical extraction order:
1. Native `AIMessage.tool_calls`
2. `additional_kwargs["tool_calls"]`
3. XML-based `<tool_call>...</tool_call>`
4. JSON-in-text fallback `tool_uses`

Canonical normalized call shape:
`{"id": str, "name": str, "args": dict}`

## Consequences
- Single execution path for executor/catcher.
- Better portability across model providers.
- Fewer silent failures where plans appear in text but tools are not executed.
