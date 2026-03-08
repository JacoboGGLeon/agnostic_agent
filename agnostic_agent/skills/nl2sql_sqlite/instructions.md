# NL2SQL SQLite Skill

Use this skill when the user wants to query a SQLite database in natural language.

Primary goal:
- Generate a SQL `SELECT` statement from the real schema of the target DB.
- Execute in read-only mode.
- Return a user-friendly answer backed by query results.

Rules:
1. Prefer `nl2sql_sqlite` tool for schema-aware SQL generation.
2. Always provide `db_path` when possible (for example DBs under `session/`).
3. Keep queries read-only (`SELECT` only).
4. Respect `row_limit` and avoid huge responses.
5. If query cannot be answered safely, explain why and ask for clarification.

Input expectations:
- A natural-language question.
- Optional `db_path` to the SQLite file.
- Optional `row_limit` and `execute` behavior.

Output expectations:
- Generated SQL.
- Execution status.
- Rows and row count when execution is enabled.
- Short explanation in plain language.
