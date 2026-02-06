from __future__ import annotations

"""
Contexto externo y Knowledge Bases para el Agnostic Deep Agent 2026.

Este módulo define:
- KnowledgeBase: descriptor ligero de una KB (faostat, sql, vector, api, etc.).
- get_default_context(): lista base de KBs disponibles (opcionalmente desde setup.yaml).
- get_kb_by_names(): helper para filtrar KBs según AgentInput.kb_names.
- build_kb_from_paths(): helper para crear KBs a partir de rutas (CSV, SQLite, etc.).
- build_kb_from_setup(): helper para mapear la sección `knowledge_bases`
  de setup.yaml a objetos KnowledgeBase.

Casos típicos:
- kind="sqlite"      → BD tabular (SQLite clásica).
- kind="sqlite-vec"  → VDB vectorial sobre sqlite-vec.
- kind="table"       → tabla externa (CSV, parquet) tratada como KB tabular.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterable
from pathlib import Path


__all__ = [
    "KnowledgeBase",
    "build_kb_from_paths",
    "build_kb_from_setup",
    "get_default_context",
    "get_kb_by_names",
]


@dataclass
class KnowledgeBase:
    """
    Descriptor simple de una KB tabular/vectorial/etc.

    - name: identificador único de la KB (ej. 'FAOSTAT_WORLD').
    - kind: tipo de backend (ej. 'sqlite', 'sqlite-vec', 'table', 'api', ...).
    - config: parámetros para conectar (rutas, DSNs, tablas, índices, etc.).

      Convenciones suaves para `config` (no obligatorias, pero recomendadas):
      - "path": ruta a archivo o DB (CSV, .db, parquet, etc.).
      - "table": nombre de tabla (si aplica, p.ej. en SQLite).
      - "role": rol lógico ('param_table', 'rules_table', 'dictionary', ...).
    """
    name: str
    kind: str = "generic"
    config: Dict[str, Any] = field(default_factory=dict)

    # Helpers de conveniencia (no obligatorios)
    @property
    def path(self) -> Optional[str]:
        """Ruta principal asociada a la KB (si existe en config)."""
        return self.config.get("path")

    @property
    def table(self) -> Optional[str]:
        """Nombre de tabla asociado (si aplica en backends tabulares)."""
        return self.config.get("table")

    @property
    def role(self) -> Optional[str]:
        """Rol lógico de la KB (param_table, rules_table, dictionary, ...)."""
        return self.config.get("role")


# ─────────────────────────────────────────────
# Builders de KB
# ─────────────────────────────────────────────

def build_kb_from_paths(
    paths: Iterable[str],
    *,
    kind: str = "table",
    default_role: Optional[str] = None,
) -> List[KnowledgeBase]:
    """
    Crea KnowledgeBase a partir de una lista de rutas (CSV, parquet, etc.).

    Pensado para casos como:
      context = ["parametrias.csv", "diccionario_abreviaturas.csv"]

    Ejemplo:
        kbs = build_kb_from_paths(
            ["parametrias.csv", "diccionario.csv"],
            kind="table",
        )

    Cada KB tendrá:
      - name: nombre derivado del stem del archivo (MAYÚSCULAS).
      - kind: el proporcionado (por defecto "table").
      - config: {"path": <ruta>, "role": default_role} si se pasa.
    """
    kb_list: List[KnowledgeBase] = []
    for p in paths:
        if not p:
            continue
        path_obj = Path(p)
        name = path_obj.stem.upper()
        cfg: Dict[str, Any] = {"path": str(path_obj)}
        if default_role is not None:
            cfg["role"] = default_role
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
