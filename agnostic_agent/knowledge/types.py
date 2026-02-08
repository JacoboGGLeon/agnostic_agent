from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# FROM context.py (Knowledge Source Config)
# ─────────────────────────────────────────────

@dataclass
class KnowledgeBase:
    """
    Representa una fuente de conocimiento disponible para el agente.
    """
    name: str                # ID único (ej. "PARAM_TABLE", "FAOSTAT_DB")
    kind: str                # Tipo (ej. "table", "sqlite", "sqlite-vec", "generic")
    config: Dict[str, Any]   # Configuración específica (path, connection_string, etc.)
    description: Optional[str] = None # Descripción semántica para el Planner


# ─────────────────────────────────────────────
# FROM knowledge_offline.py (Vector/PDF Models)
# ─────────────────────────────────────────────

class ElementNode(BaseModel):
    id: str
    page: int
    kind: str
    md: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    prev_id: Optional[str] = None
    next_id: Optional[str] = None
    source_path: str

class Chunk(BaseModel):
    chunk_id: str
    element_id: str
    page: int
    md: str
    neighbor_ids: List[str] = Field(default_factory=list)
    source_path: str
