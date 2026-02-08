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
        kb_list.append(
            KnowledgeBase(
                name=str(name),
                kind=str(kind),
                config=config,
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
