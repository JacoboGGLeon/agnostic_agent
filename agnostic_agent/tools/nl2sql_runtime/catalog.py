from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_catalog(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def catalog_items(catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    schemas = catalog.get("schemas") if isinstance(catalog.get("schemas"), dict) else {}
    relationships = catalog.get("relationships") if isinstance(catalog.get("relationships"), list) else []
    glossary = catalog.get("business_glossary") if isinstance(catalog.get("business_glossary"), dict) else {}
    items: List[Dict[str, Any]] = []
    for schema_name, schema_obj in schemas.items():
        tables = schema_obj.get("tables") if isinstance(schema_obj, dict) else {}
        if not isinstance(tables, dict):
            continue
        for table_name, table_meta in tables.items():
            if not isinstance(table_meta, dict):
                continue
            items.append(
                {
                    "type": "table",
                    "schema": schema_name,
                    "table": table_name,
                    "description": str(table_meta.get("description", "")),
                    "planner_context": str(table_meta.get("description", "")),
                    "rich_context": table_meta,
                    "business_glossary": glossary,
                }
            )
            for col in table_meta.get("columns", []) or []:
                if not isinstance(col, dict):
                    continue
                items.append(
                    {
                        "type": "column",
                        "schema": schema_name,
                        "table": table_name,
                        "column": str(col.get("name", "")),
                        "description": str(col.get("description", "")),
                        "planner_context": f"{table_name}.{col.get('name', '')} {col.get('description', '')}",
                        "rich_context": col,
                        "business_glossary": glossary,
                    }
                )
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        items.append(
            {
                "type": "join",
                "schema": str(rel.get("left_schema") or "main"),
                "table": str(rel.get("left_table", "")),
                "column": str(rel.get("left_column", "")),
                "description": str(rel.get("description", "")),
                "planner_context": (
                    f"{rel.get('left_schema', 'main')}.{rel.get('left_table', '')}.{rel.get('left_column', '')} "
                    f"{rel.get('right_schema', 'main')}.{rel.get('right_table', '')}.{rel.get('right_column', '')} "
                    f"{rel.get('description', '')}"
                ),
                "rich_context": rel,
                "business_glossary": glossary,
            }
        )
    return items
