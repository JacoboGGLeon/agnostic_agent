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
    text: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None
    prev_id: Optional[str] = None
    next_id: Optional[str] = None
    source_path: str
    is_boilerplate: bool = False
    section_path: Optional[str] = None


class ChunkLocator(BaseModel):
    source_path: str
    page_start: int
    page_end: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    section_path: Optional[str] = None

class ChunkContent(BaseModel):
    text: str
    text_normalized: str
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    content_type: str
    language: str = "es"

class ChunkTags(BaseModel):
    document_type: str = "document"

class ChunkQuality(BaseModel):
    is_boilerplate: bool = False
    embed_model: str = "default"
    token_count_estimated: int = 0

class Chunk(BaseModel):
    doc_id: str
    chunk_pk: str
    locator: ChunkLocator
    content: ChunkContent
    tags: ChunkTags
    quality: ChunkQuality
