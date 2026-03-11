"""Local wrapper declarations for chat_db."""


def tool_contracts():
    return {
        "skill": "chat_db",
        "tools": ["nl2sql", "inspect_sqlite_schema", "execute_sql_readonly"],
    }
