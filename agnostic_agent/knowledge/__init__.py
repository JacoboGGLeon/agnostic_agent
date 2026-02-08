from __future__ import annotations

"""
Gestión del CONTEXTO y CONOCIMIENTO para el Agnostic Deep Agent 2026.
REGISTRY REFACTORIZADO.

Reemplaza al antiguo `context.py`.
Gestiona las fuentes de conocimiento (KnowledgeBase) y sus conectores.
"""

from typing import List, Dict, Any, Optional
from agnostic_agent.knowledge.types import KnowledgeBase

# Expose key modules for easy access
# from agnostic_agent.knowledge import vector, sql, tabular, json_store

def build_kb_from_paths(
    file_paths: List[str],
    kind: str = "table",
    role_prefix: str = "custom",
) -> List[KnowledgeBase]:
    """
    Crea objetos KnowledgeBase al vuelo a partir de una lista de archivos.
    Útil para context_files inyectados en runtime.
    """
    kb_list = []
    for idx, path in enumerate(file_paths):
        name = f"{role_prefix.upper()}_{idx}"
        cfg = {"path": path}
        kb_list.append(
            KnowledgeBase(
                name=name,
                kind=kind,
                config=cfg,
            )
        )
    return kb_list


def build_kb_from_setup(setup_cfg: Dict[str, Any]) -> List[KnowledgeBase]:
    """
    Construye una lista de KnowledgeBase a partir de la sección `knowledge_bases`
    de setup.yaml.
    """
    kb_section = setup_cfg.get("knowledge_bases") or []
    kb_list: List[KnowledgeBase] = []

    # Normalizamos a lista de dicts
    if isinstance(kb_section, dict):
        items: List[Dict[str, Any]] = []
        for name, cfg in kb_section.items():
            if not isinstance(cfg, dict):
                continue
            item = dict(cfg)
            item.setdefault("name", name)
            items.append(item)
    elif isinstance(kb_section, list):
        items = [x for x in kb_section if isinstance(x, dict)]
    else:
        items = []

    for item in items:
        name = item.get("name")
        if not name:
            continue
        kind = item.get("kind", "generic")
        config = item.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        
        description = item.get("description")
        
        # Auto-enrich description for sqlite-vec if missing
        if not description and kind == "sqlite-vec":
            db_path = config.get("path")
            if db_path:
                import os
                if os.path.exists(db_path):
                    try:
                        from agnostic_agent.knowledge.vector import get_kb_description_from_db
                        description = get_kb_description_from_db(db_path)
                    except ImportError:
                        pass
        
        kb_list.append(
            KnowledgeBase(
                name=str(name),
                kind=str(kind),
                config=config,
                description=description,
            )
        )

    return kb_list


def get_default_context(
    setup_cfg: Optional[Dict[str, Any]] = None,
) -> List[KnowledgeBase]:
    """
    Devuelve la lista de KBs disponibles por defecto.
    """
    if setup_cfg:
        kbs = build_kb_from_setup(setup_cfg)
        if kbs:
            return kbs
            
    # Fallback: check standard 'embeddings.db'
    import os
    default_db = "embeddings.db"
    
    # Try looking in typical places
    candidates = [
        default_db,
        os.path.join(os.getcwd(), default_db),
    ]
    
    found_db = None
    for c in candidates:
        if os.path.exists(c):
            found_db = c
            break
            
    if found_db:
        desc = ""
        try:
            from agnostic_agent.knowledge.vector import get_kb_description_from_db
            desc = get_kb_description_from_db(found_db)
        except ImportError:
            pass
            
        return [
            KnowledgeBase(
                name="knowledge_base",
                kind="sqlite-vec",
                config={"path": found_db},
                description=desc
            )
        ]
        
    return []


def get_kb_by_names(
    kb_names: List[str],
    all_kb: Optional[List[KnowledgeBase]] = None,
) -> List[KnowledgeBase]:
    """
    Devuelve las KnowledgeBase cuyo name esté en kb_names.
    """
    kb_list = all_kb if all_kb is not None else get_default_context()
    if not kb_names:
        return kb_list

    name_set = set(kb_names)
    return [kb for kb in kb_list if kb.name in name_set]
