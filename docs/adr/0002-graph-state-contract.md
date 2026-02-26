# ADR 0002 - Graph State Contract

## Status
Accepted

## Context
`logic.py` coordinates analyzer/planner/executor/catcher/summarizer/validator using a shared `State`.
Frequent regressions happen when nodes mutate fields inconsistently.

## Decision
- `State` in `agnostic_agent.logic` is the canonical graph contract.
- Node-level rules:
  - `analyzer` owns `analyzer`.
  - `planner` owns `planner_trajs` and planning tool calls.
  - `executor` owns `executor_steps` and tool messages.
  - `catcher` owns `tool_runs`.
  - `summarizer` owns `summary/pipeline_summary/dev_out/deep_out/user_out`.
  - `validator` may annotate `validator` and patch `user_out` only as guardrail.

## Consequences
- Reduced ambiguity around field ownership.
- Easier extraction of `logic.py` into smaller modules.
- Better compatibility with LangGraph Graph API best practices.
