# ADR 0001 - Config Source of Truth

## Status
Accepted

## Context
There are two config module paths in the package tree:
- `agnostic_agent/config/*` (canonical)
- `agnostic_agent/agnostic_agent/config/*` (duplicate compatibility path)

Keeping both with independent logic causes drift and hard-to-debug behavior.

## Decision
- `agnostic_agent/config/*` is the single source of truth.
- `agnostic_agent/agnostic_agent/config/*` remains as compatibility wrappers only.
- New config changes must be implemented only in canonical modules.

## Consequences
- No behavior drift between duplicate modules.
- Legacy imports keep working.
- Future cleanup can remove wrappers in a major version.
