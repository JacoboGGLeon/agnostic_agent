from __future__ import annotations

"""
Gestión del CONTEXTO para el Agnostic Deep Agent 2026.

Define:
- KnowledgeBase: dataclass con la info de una fuente de conocimiento.
- get_default_context: recupera KBs por defecto o desde un dict (setup.yaml).
- Helpers para filtrar y construir KBs.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class KnowledgeBase:
    """
    Representa una fuente de conocimiento disponible para el agente.
    """
    name: str                # ID único (ej. "PARAM_TABLE", "FAOSTAT_DB")
    kind: str                # Tipo (ej. "table", "sqlite", "sqlite-vec", "generic")
    config: Dict[str, Any]   # Configuración específica (path, connection_string, etc.)


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
    de setup.yaml ya cargado en un dict.

    Forma esperada (tolerante):

    knowledge_bases:
      - name: PARAM_TABLE
        kind: table
        config:
          path: /content/parametrias.csv
          role: param_table

      - name: RULES_TABLE
        kind: table
        config:
          path: /content/reglas_calidad.csv
          role: rules_table

      - name: FAOSTAT_SQLITE
        kind: sqlite
        config:
          db_path: /content/faostat_world.db

    También acepta formato dict:

    knowledge_bases:
      PARAM_TABLE:
        kind: table
        config:
          path: /content/parametrias.csv
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


# ─────────────────────────────────────────────
# Contexto por defecto
# ─────────────────────────────────────────────

def get_default_context(
    setup_cfg: Optional[Dict[str, Any]] = None,
) -> List[KnowledgeBase]:
    """
    Devuelve la lista de KBs disponibles por defecto.

    Modos de uso:

      # 1) Sin setup.yaml (modo antiguo / hardcoded)
      kbs = get_default_context()

      # 2) Pasando el dict de setup.yaml ya cargado
      from pathlib import Path
      import yaml

      with Path("setup.yaml").open("r", encoding="utf-8") as f:
          cfg = yaml.safe_load(f) or {}

      kbs = get_default_context(cfg)

    Si `setup_cfg` se pasa y contiene `knowledge_bases`,
    se usa `build_kb_from_setup(setup_cfg)`.

    Si no, se devuelve una lista vacía (o podrías hardcodear aquí KBs
    globales tipo FAOSTAT_SQLITE/FAOSTAT_VECTOR).
    """
    if setup_cfg:
        kbs = build_kb_from_setup(setup_cfg)
        if kbs:
            return kbs

    # Ejemplo de hardcode si quisieras algo global:
    # return [
    #     KnowledgeBase(
    #         name="FAOSTAT_SQLITE",
    #         kind="sqlite",
    #         config={"db_path": "/content/faostat_world.db"},
    #     ),
    #     KnowledgeBase(
    #         name="FAOSTAT_VECTOR",
    #         kind="sqlite-vec",
    #         config={
    #             "db_path": "/content/faostat_world.db",
    #             "table": "faostat_vec",
    #         },
    #     ),
    # ]
    return []


# ─────────────────────────────────────────────
# Filtro por nombres
# ─────────────────────────────────────────────

def get_kb_by_names(
    kb_names: List[str],
    all_kb: Optional[List[KnowledgeBase]] = None,
) -> List[KnowledgeBase]:
    """
    Devuelve las KnowledgeBase cuyo name esté en kb_names.

    - kb_names: nombres solicitados (p.ej. AgentInput.kb_names).
    - all_kb:  lista total de KBs disponibles; si es None, usamos get_default_context().

    Si kb_names está vacío, se devuelve all_kb completa.
    """
    kb_list = all_kb if all_kb is not None else get_default_context()
    if not kb_names:
        return kb_list

    name_set = set(kb_names)
    return [kb for kb in kb_list if kb.name in name_set]
