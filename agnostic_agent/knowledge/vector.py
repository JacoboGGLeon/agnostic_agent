from __future__ import annotations

import os
import json
import hashlib
import sqlite3
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union, Callable

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Import shared types
from agnostic_agent.knowledge.types import (
    ElementNode, Chunk, ChunkLocator, ChunkContent, ChunkTags, ChunkQuality
)

# Try to import fitz (PyMuPDF)
try:
    import fitz
except ImportError:
    fitz = None

# Try to import docling
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

from agnostic_agent.capabilities import DEFAULT_EMB_ID, LocalModelPaths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMB_MODEL_REPO = os.getenv("EMB_MODEL_ID", DEFAULT_EMB_ID)
MODELS_CACHE_DIR = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
EMB_DIM = 1024 # Context dependent, but keeping default for now


# -----------------------------------------------------------------------------
# PDF Parsing Logic
# -----------------------------------------------------------------------------

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def _parse_with_docling(pdf_path: str) -> Tuple[List[ElementNode], int]:
    if DocumentConverter is None:
        logger.warning("Docling not installed. Skipping docling parser.")
        return [], 0
    
    try:
        conv = DocumentConverter()
        doc = conv.convert(pdf_path)
    except Exception as e:
        logger.error(f"Docling conversion failed for {pdf_path}: {e}")
        return [], 0

    # Handle different docling versions/attributes
    pages = getattr(doc, "pages", []) or []
    if not pages and hasattr(doc, "document"): # Try nested document object if present
         pages = getattr(doc.document, "pages", [])

    total_pages = len(pages)
    nodes: List[ElementNode] = []
    node_seq = 0
    
    iterable_pages = pages.values() if isinstance(pages, dict) else pages

    for page_idx, page in enumerate(iterable_pages, start=1):
        candidates = []
        for attr in ["items", "elements", "blocks", "cells"]:
            if hasattr(page, attr):
                arr = getattr(page, attr)
                if isinstance(arr, list) and len(arr) > 0:
                    candidates = arr
                    break
        
        for it in candidates:
            # Extract text/markdown
            md = ""
            try:
                if hasattr(it, "export_to_markdown"):
                    md = it.export_to_markdown()
                elif hasattr(it, "to_markdown"):
                    md = it.to_markdown()
                elif hasattr(it, "text"):
                    md = it.text
            except Exception:
                md = getattr(it, "text", "") or ""
            
            text = getattr(it, "text", "") or md
            if not text.strip():
                continue
                
            kind = it.__class__.__name__.lower()
            label = str(getattr(it, "label", getattr(it, "type", ""))).lower()
            is_boilerplate = "header" in label or "footer" in label

            bbox = getattr(it, "bbox", None)
            # Ensure bbox is tuple if present
            if bbox and hasattr(bbox, "as_tuple"):
                bbox = bbox.as_tuple()
            
            node_id = f"{Path(pdf_path).name}::p{page_idx}::{node_seq}::{sha1(text)[:6]}"
            nodes.append(ElementNode(
                id=node_id, 
                page=page_idx, 
                kind=kind, 
                md=md, 
                text=text,
                bbox=bbox, 
                source_path=str(pdf_path),
                is_boilerplate=is_boilerplate
            ))
            node_seq += 1

    return _link_nodes(nodes), total_pages

def _parse_with_pymupdf(pdf_path: str) -> Tuple[List[ElementNode], int]:
    if fitz is None:
        logger.warning(f"PyMuPDF (fitz) not installed. Skipping fallback for {pdf_path}.")
        return [], 0
        
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"PyMuPDF open failed for {pdf_path}: {e}")
        return [], 0

    total_pages = len(doc)
    nodes: List[ElementNode] = []
    node_seq = 0
    
    logger.info(f"PyMuPDF: Processing {total_pages} pages in {pdf_path}")
    
    for page_idx in range(total_pages):
        page = doc[page_idx]
        blocks = page.get_text("blocks") # (x0, y0, x1, y1, text, block_no, block_type)
        
        if not blocks:
            logger.info(f"PyMuPDF: Page {page_idx+1} has no text blocks.")
            continue

        for b in blocks:
            if len(b) < 5:
                continue
            x0, y0, x1, y1, text = b[:5]
            text = (text or "").strip()
            if not text:
                continue
            
            kind = "paragraph"
            # Heuristic: top margins and bottom margins of A4/Letter size usually map to headers/footers
            is_boilerplate = bool(y0 < 60 or y1 > 730)

            node_id = f"{Path(pdf_path).name}::p{page_idx+1}::{node_seq}::{sha1(text)[:6]}"
            nodes.append(ElementNode(
                id=node_id, 
                page=page_idx+1, 
                kind=kind, 
                md=text, 
                text=text,
                bbox=(x0, y0, x1, y1), 
                source_path=str(pdf_path),
                is_boilerplate=is_boilerplate
            ))
            node_seq += 1
            
    logger.info(f"PyMuPDF: Extracted {len(nodes)} nodes.")
    return _link_nodes(nodes), total_pages

def _link_nodes(nodes: List[ElementNode]) -> List[ElementNode]:
    """Helper to link prev/next IDs per page."""
    by_page: Dict[int, List[ElementNode]] = {}
    for n in nodes:
        by_page.setdefault(n.page, []).append(n)
    
    for page_nodes in by_page.values():
        for i, n in enumerate(page_nodes):
            n.prev_id = page_nodes[i-1].id if i > 0 else None
            n.next_id = page_nodes[i+1].id if i < len(page_nodes)-1 else None
    return nodes

def parse_pdf(pdf_path: str) -> Tuple[List[ElementNode], int]:
    """Tries Docling first, falls back to PyMuPDF."""
    nodes = []
    total_pages = 0
    
    # 1. Try Docling
    if DocumentConverter:
        logger.info(f"Using Docling parser for {pdf_path}")
        nodes, total_pages = _parse_with_docling(pdf_path)
    else:
        logger.info("Docling not available.")

    # 2. Fallback to PyMuPDF if Docling failed or returned no nodes
    if not nodes:
        logger.info(f"Docling returned no nodes (or not avail). Falling back to PyMuPDF for {pdf_path}.")
        nodes, total_pages = _parse_with_pymupdf(pdf_path)
    
    if not nodes:
        logger.error(f"Failed to extract text from {pdf_path} with both Docling and PyMuPDF.")
        
    return nodes, total_pages


# -----------------------------------------------------------------------------
# Chunking Logic
# -----------------------------------------------------------------------------

def build_chunks(nodes: List[ElementNode], k_neighbors: int = 1) -> List[Chunk]:
    if not nodes:
        return []
        
    source_path = nodes[0].source_path
    doc_id = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:16]

    chunks: List[Chunk] = []
    by_page: Dict[int, List[ElementNode]] = {}
    for n in nodes:
        by_page.setdefault(n.page, []).append(n)

    for page, arr in by_page.items():
        for i, n in enumerate(arr):
            left = max(0, i - k_neighbors)
            right = min(len(arr), i + k_neighbors + 1)
            
            neigh_before = [x.text for j, x in enumerate(arr[left:i])]
            neigh_after = [x.text for j, x in enumerate(arr[i+1:right])]

            text_norm = n.text.lower().strip()
            chunk_pk = sha1(text_norm + doc_id)[:10]

            locator = ChunkLocator(
                source_path=n.source_path,
                page_start=n.page,
                page_end=n.page,
                bbox=n.bbox,
                section_path=n.section_path
            )

            content = ChunkContent(
                text=n.text,
                text_normalized=text_norm,
                context_before="\n".join(neigh_before) if neigh_before else None,
                context_after="\n".join(neigh_after) if neigh_after else None,
                content_type=n.kind,
                language="es"
            )

            tags = ChunkTags(document_type="document")
            
            quality = ChunkQuality(
                is_boilerplate=n.is_boilerplate,
                embed_model=EMB_MODEL_REPO,
                token_count_estimated=len(n.text.split())
            )

            chunks.append(Chunk(
                doc_id=doc_id,
                chunk_pk=chunk_pk,
                locator=locator,
                content=content,
                tags=tags,
                quality=quality
            ))
    return chunks


# -----------------------------------------------------------------------------
# Embedding Logic
# -----------------------------------------------------------------------------

_EMBEDDER_CACHE: Dict[str, Any] = {}

def get_vllm_client():
    from openai import OpenAI
    api_base = os.getenv("VLLM_EMB_API_BASE") or os.getenv("VLLM_EMB_URL", "http://localhost:8001/v1")
    api_key = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY", "EMPTY")
    return OpenAI(base_url=api_base, api_key=api_key)

def check_vllm_embedding_available() -> bool:
    """Checks if vLLM embedding endpoint is responsive."""
    try:
        client = get_vllm_client()
        client.embeddings.create(input=["test"], model=EMB_MODEL_REPO)
        logger.info(f"vLLM embedding endpoint found at {client.base_url}")
        return True
    except Exception:
        logger.info("vLLM embedding endpoint not found or error. prompting local fallback.")
        return False

def get_embedder():
    """Singleton-ish loader for the embedding model."""
    if "tokenizer" in _EMBEDDER_CACHE and "model" in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE["tokenizer"], _EMBEDDER_CACHE["model"]

    logger.info(f"Loading embedding model (LOCAL): {EMB_MODEL_REPO}")
    
    device = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu")
    logger.info(f"Using device '{device}' for local embeddings (default: cpu).")

    try:
        tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_REPO, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            EMB_MODEL_REPO, 
            trust_remote_code=True, 
            device_map=device,
            torch_dtype=torch.float32 # CPU usually needs float32
        )
    except Exception as e:
        logger.error(f"Failed to load model from HF {EMB_MODEL_REPO}: {e}")
        raise e

    _EMBEDDER_CACHE["tokenizer"] = tokenizer
    _EMBEDDER_CACHE["model"] = model
    return tokenizer, model

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(summed / counts, dim=1)

@torch.inference_mode()
def embed_texts(texts: List[str], batch_size: int = 8) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMB_DIM), dtype="float32")

    use_vllm = os.getenv("USE_VLLM_EMBEDDING", "0") == "1"
    
    if use_vllm:
        try:
            client = get_vllm_client()
            all_vecs = []
            requested_dimensions = os.getenv("OPENAI_EMBED_DIMENSIONS", "").strip()
            embed_kwargs: Dict[str, Any] = {}
            if requested_dimensions.isdigit():
                embed_kwargs["dimensions"] = int(requested_dimensions)
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch = [t.replace("\n", " ") for t in batch] 
                
                resp = client.embeddings.create(
                    input=batch,
                    model=EMB_MODEL_REPO,
                    **embed_kwargs,
                )
                vecs = [d.embedding for d in resp.data]
                all_vecs.append(np.array(vecs, dtype="float32"))
            
            return np.vstack(all_vecs)
        except Exception as e:
             logger.warning(f"vLLM embedding failed ({e}), falling back to local Transformers.")
    
    tokenizer, model = get_embedder()
    all_vecs = []
    
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=8192, 
            return_tensors="pt"
        ).to(model.device)
        
        out = model(**inputs)
        vecs = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
        all_vecs.append(vecs.float().cpu().numpy())
    
    return np.vstack(all_vecs)


# -----------------------------------------------------------------------------
# Database Logic (SQLite + sqlite-vec)
# -----------------------------------------------------------------------------

def init_db(db_path: str):
    """Initializes the SQLite database with sqlite-vec extension."""
    import sqlite_vec
    
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS v_chunks
        USING vec0(embedding FLOAT[{EMB_DIM}]);
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks_meta (
            rowid INTEGER PRIMARY KEY,
            chunk_pk TEXT UNIQUE,
            doc_id TEXT,
            source_path TEXT,
            locator TEXT,
            content TEXT,
            tags TEXT,
            quality TEXT
        );
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files_meta (
            source_path TEXT PRIMARY KEY,
            description TEXT,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # L2 index: one embedding centroid per document (source_path).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS docs_index (
            source_path TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            n_chunks INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_meta_source_path ON chunks_meta(source_path);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_meta_source_page ON chunks_meta(source_path, json_extract(locator, '$.page_start'));"
    )
    conn.commit()
    conn.close()

def _pack_f32(arr: np.ndarray) -> bytes:
    return arr.astype("float32").tobytes()


def _unpack_f32(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec) + 1e-9
    return (vec / denom).astype("float32")


def _doc_centroid(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((EMB_DIM,), dtype="float32")
    return _normalize(np.mean(embeddings, axis=0))


def _tokenize_lex(text: str) -> List[str]:
    import re as _re
    cleaned = _re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return [t for t in cleaned.split() if len(t) >= 2]


def _lexical_overlap_score(query: str, descriptor: str) -> float:
    q_toks = _tokenize_lex(query)
    if not q_toks:
        return 0.0
    d = (descriptor or "").lower()
    hits = sum(1 for t in q_toks if t in d)
    return float(hits) / float(len(q_toks))


def upsert_document_index(db_path: str, source_path: str, chunk_embeddings: np.ndarray, description: str = "") -> None:
    """
    Maintains L2 document-level index. En lugar de usar el centroide borroso de los chunks,
    ahora crea un vector semántico fuerte basado en la DESCRIPCIÓN y el TÍTULO del documento.
    """
    import sqlite_vec

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    text_to_embed = f"{Path(source_path).name}. {description}".strip()
    try:
        l2_vec = embed_texts([text_to_embed])[0]
    except Exception:
        l2_vec = _doc_centroid(chunk_embeddings)

    blob = _pack_f32(l2_vec)
    n_chunks = int(chunk_embeddings.shape[0]) if chunk_embeddings.ndim > 1 else int(bool(chunk_embeddings.size))

    conn.execute(
        """
        INSERT OR REPLACE INTO docs_index (source_path, embedding, n_chunks, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (str(source_path), blob, n_chunks),
    )
    conn.commit()
    conn.close()

def upsert_chunks(db_path: str, chunks: List[Chunk], embeddings: np.ndarray):
    """Inserts chunks and their embeddings into the DB."""
    import sqlite_vec

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    cur = conn.cursor()
    
    for i, ch in enumerate(chunks):
        blob = _pack_f32(embeddings[i])
        
        cur.execute("""
            INSERT OR REPLACE INTO chunks_meta(chunk_pk, doc_id, source_path, locator, content, tags, quality)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            ch.chunk_pk, 
            ch.doc_id, 
            ch.locator.source_path,
            ch.locator.model_dump_json(), 
            ch.content.model_dump_json(), 
            ch.tags.model_dump_json(), 
            ch.quality.model_dump_json()
        ))
        
        cur.execute("SELECT rowid FROM chunks_meta WHERE chunk_pk = ?", (ch.chunk_pk,))
        row = cur.fetchone()
        if row:
            row_id = row[0]
            cur.execute("""
                INSERT OR REPLACE INTO v_chunks(rowid, embedding)
                VALUES (?, ?);
            """, (row_id, blob))
            
    conn.commit()
    conn.close()


def search_db(db_path: str, query: str, top_k: int = 5, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Hierarchical semantic search:
      L2 -> choose best document(s) from docs_index
      L1 -> search best chunks only inside selected document(s)
    """
    if not os.path.exists(db_path):
        return []

    q_vec = embed_texts([query])[0]
    q_blob = _pack_f32(q_vec)

    import sqlite_vec
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    try:
        # Strict mode: require hierarchical index already materialized.
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        required = {"v_chunks", "chunks_meta", "docs_index"}
        if not required.issubset(tables):
            logger.error(
                "search_db requires strict HKB schema (chunks_meta + docs_index). "
                "Reingest documents with the current pipeline."
            )
            return []

        q_vec_norm = _normalize(q_vec)
        desc_map: Dict[str, str] = {}
        try:
            for s_path, desc in conn.execute(
                "SELECT source_path, description FROM files_meta"
            ).fetchall():
                desc_map[str(s_path)] = str(desc or "")
        except Exception:
            desc_map = {}

        # Resolve candidate documents (L2)
        if source_filter:
            sf = source_filter.strip()
            doc_rows = conn.execute(
                """
                SELECT source_path, embedding
                FROM docs_index
                WHERE source_path = ?
                   OR source_path LIKE ?
                   OR source_path LIKE ?
                """,
                (sf, f"%/{sf}", f"%\\{sf}"),
            ).fetchall()
            doc_candidates = []
            for s_path, emb_blob in doc_rows:
                d_vec = _unpack_f32(emb_blob)
                vec_score = float(np.dot(q_vec_norm, _normalize(d_vec)))
                descriptor = f"{Path(s_path).name} {desc_map.get(str(s_path), '')}"
                lex_score = _lexical_overlap_score(query, descriptor)
                # Since d_vec is now the semantic embedding of the description, it is highly accurate.
                d_score = (0.75 * max(0.0, vec_score)) + (0.25 * lex_score)
                doc_candidates.append((s_path, d_score))
            doc_candidates.sort(key=lambda x: x[1], reverse=True)
        else:
            doc_rows = conn.execute(
                "SELECT source_path, embedding FROM docs_index"
            ).fetchall()

            doc_candidates = []
            for s_path, emb_blob in doc_rows:
                d_vec = _unpack_f32(emb_blob)
                vec_score = float(np.dot(q_vec_norm, _normalize(d_vec)))
                descriptor = f"{Path(s_path).name} {desc_map.get(str(s_path), '')}"
                lex_score = _lexical_overlap_score(query, descriptor)
                # Since d_vec is now the semantic embedding of the description, it is highly accurate.
                d_score = (0.75 * max(0.0, vec_score)) + (0.25 * lex_score)
                doc_candidates.append((s_path, d_score))

            doc_candidates.sort(key=lambda x: x[1], reverse=True)

            # Expansion strategy: top-1 anchor + neighbors close in score.
            if doc_candidates:
                best = doc_candidates[0][1]
                expanded = [c for c in doc_candidates if c[1] >= (best - 0.08)]
                doc_candidates = expanded[:3] if len(expanded) > 3 else expanded

        if not doc_candidates:
            return []

        # L1 search inside selected L2 documents.
        l1_fetch_k = max(10, top_k * 3)
        all_results = []
        seen_chunk_ids = set()

        for doc_path, doc_score in doc_candidates:
            rows = conn.execute(
                """
                SELECT
                    v.distance,
                    m.chunk_pk,
                    m.doc_id,
                    m.source_path,
                    m.locator,
                    m.content,
                    m.tags,
                    m.quality
                FROM v_chunks v
                JOIN chunks_meta m ON m.rowid = v.rowid
                WHERE v.embedding MATCH ?
                  AND k = ?
                  AND m.source_path = ?
                  AND json_extract(m.quality, '$.is_boilerplate') = 0
                ORDER BY v.distance ASC;
                """,
                (q_blob, l1_fetch_k, doc_path),
            ).fetchall()

            for dist, chunk_pk, doc_id, source_path, loc_raw, content_raw, tags_raw, quality_raw in rows:
                chunk_sim = max(0.0, min(1.0, 1.0 - (dist**2) / 2.0))
                fused_score = (0.75 * chunk_sim) + (0.25 * max(0.0, doc_score))
                if chunk_pk in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_pk)
                
                # Late-binding context
                content_dict = json.loads(content_raw)
                main_text = content_dict.get("text", "")
                ctx_before = content_dict.get("context_before")
                ctx_after = content_dict.get("context_after")
                
                parts = []
                if ctx_before: parts.append(ctx_before)
                parts.append(main_text)
                if ctx_after: parts.append(ctx_after)
                
                bound_text = "\n".join(parts)
                locator_dict = json.loads(loc_raw)

                all_results.append(
                    {
                        "score": float(fused_score),
                        "distance": float(dist),
                        "chunk_score": float(chunk_sim),
                        "doc_score": float(max(0.0, doc_score)),
                        "doc_descriptor": f"{Path(source_path).name} {desc_map.get(str(source_path), '')}".strip(),
                        "chunk_id": chunk_pk,
                        "element_id": chunk_pk,
                        "page": locator_dict.get("page_start", 0),
                        "md": bound_text,
                        "neighbors": [],
                        "source_path": source_path,
                        "search_tree": "L2->L1",
                        "content_metadata": content_dict,
                        "locator_metadata": locator_dict
                    }
                )

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []
    finally:
        conn.close()

def get_stats(db_path: str) -> Dict[str, Any]:
    if not os.path.exists(db_path):
        return {"chunks": 0, "files": 0, "size_bytes": 0, "vector_count": 0, "dim": 0}
        
    size_bytes = os.path.getsize(db_path)
    conn = sqlite3.connect(db_path)
    
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as e:
        logger.warning(f"Could not load sqlite-vec in get_stats: {e}")

    try:
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
        n_files = conn.execute("SELECT COUNT(DISTINCT source_path) FROM chunks_meta").fetchone()[0]
        
        try:
            n_vectors = conn.execute("SELECT count(*) FROM v_chunks").fetchone()[0]
        except:
            n_vectors = 0
            
        return {
            "chunks": n_chunks, 
            "files": n_files, 
            "size_bytes": size_bytes,
            "vector_count": n_vectors,
            "dim": EMB_DIM,
            "doc_index_count": conn.execute("SELECT COUNT(*) FROM docs_index").fetchone()[0]
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"chunks": 0, "files": 0, "size_bytes": size_bytes, "vector_count": 0, "dim": 0}
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Main Facade
# -----------------------------------------------------------------------------

def get_kb_description_from_db(db_path: str) -> str:
    """Returns a summary description of the KB based on ingested files."""
    if not os.path.exists(db_path):
        return ""
    
    try:
        conn = sqlite3.connect(db_path)
        # Check if table exists first to avoid error on old DBs
        try:
            rows = conn.execute("SELECT source_path, description FROM files_meta").fetchall()
        except sqlite3.OperationalError:
            # Table might not exist yet
            conn.close()
            return ""
            
        conn.close()
        
        if not rows:
            return ""
        
        lines = []
        for path, desc in rows:
            fname = Path(path).name
            if desc and desc.strip():
                lines.append(f"- {fname}: {desc}")
            else:
                lines.append(f"- {fname}")
        
        return "Contenido Ingestado:\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"Error reading KB description: {e}")
        return ""

def get_ingested_files(db_path: str) -> List[Dict[str, Any]]:
    """Returns a list of ingested files with metadata."""
    if not os.path.exists(db_path):
        return []
        
    try:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT source_path, description, ingested_at FROM files_meta ORDER BY ingested_at DESC").fetchall()
        except sqlite3.OperationalError:
            conn.close()
            return []
            
        conn.close()
        
        files = []
        for path, desc, ts in rows:
            files.append({
                "file": Path(path).name,
                "description": desc,
                "ingested_at": ts,
                "path": path
            })
        return files
    except Exception as e:
        logger.error(f"Error listing ingested files: {e}")
        return []


def get_chunks_metadata(db_path: str, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Devuelve metadata por elemento/chunk para inspección en UI.
    """
    if not os.path.exists(db_path):
        return []

    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            """
            SELECT chunk_pk, doc_id, locator, source_path, content
            FROM chunks_meta
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (int(max(1, limit)),),
        ).fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"Error reading chunks metadata: {e}")
        return []

    out: List[Dict[str, Any]] = []
    for chunk_pk, doc_id, loc_raw, source_path, content_raw in rows:
        try:
            content = json.loads(content_raw)
            md = content.get("text", "")
        except Exception:
            md = ""
            
        try:
            loc = json.loads(loc_raw)
            page = loc.get("page_start", 0)
        except Exception:
            page = 0
            
        preview = (md or "").replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:220] + "..."
        out.append(
            {
                "chunk_id": chunk_pk,
                "element_id": doc_id,
                "page": page,
                "source_path": source_path,
                "neighbors_count": 0,
                "md_preview": preview,
            }
        )
    return out

def ingest_pdf_file(
    pdf_path: str, 
    db_path: str, 
    k_neighbors: int = 1,
    description: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """High-level function to ingest a PDF."""
    if not os.path.exists(pdf_path):
        return {"error": f"File not found: {pdf_path}"}

    def _update_progress(p: float, msg: str):
        if progress_callback:
            progress_callback(p, msg)

    _update_progress(0.1, "Analyzing document structure (Parsing)...")

    # 1. Parse
    nodes, total_pages = parse_pdf(pdf_path)
    if not nodes:
        return {"error": "No text extracted from PDF."}

    _update_progress(0.3, f"Parsed {total_pages} pages. Creating chunks...")

    # 2. Chunk
    chunks = build_chunks(nodes, k_neighbors=k_neighbors)
    if not chunks:
        return {"error": "No chunks created."}

    _update_progress(0.5, f"Created {len(chunks)} chunks. Embedding...")

    # 3. Embed
    try:
        texts = [c.content.text for c in chunks]
        embeddings = embed_texts(texts)
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}

    _update_progress(0.9, "Storing vectors in database...")

    # 4. Store
    try:
        init_db(db_path) 
        upsert_chunks(db_path, chunks, embeddings)
        upsert_document_index(db_path, str(pdf_path), embeddings, description=description or "")
        
        # 4.1 Update file metadata
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT OR REPLACE INTO files_meta (source_path, description)
            VALUES (?, ?)
        """, (str(pdf_path), description or ""))
        conn.commit()
        conn.close()
        
    except Exception as e:
        return {"error": f"Database insertion failed: {e}"}
    
    _update_progress(1.0, "Ingestion complete!")

    return {
        "success": True,
        "pages": total_pages,
        "nodes": len(nodes),
        "chunks": len(chunks),
        "file": Path(pdf_path).name
    }

# -----------------------------
# Persistence / History Log
# -----------------------------
def log_ingestion_event(metadata: Dict[str, Any], history_path: str) -> None:
    import datetime
    
    if "timestamp" not in metadata:
        metadata["timestamp"] = datetime.datetime.now().isoformat()
        
    try:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")
    except Exception as e:
        logger.error(f"Error writing to history log: {e}")

def get_ingestion_history(history_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(history_path):
        return []
        
    history = []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error(f"Error reading history log: {e}")
        return []
        
    return list(reversed(history))
