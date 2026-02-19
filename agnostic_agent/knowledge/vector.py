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
from agnostic_agent.knowledge.types import ElementNode, Chunk

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
            
            if not md.strip():
                continue
                
            kind = it.__class__.__name__.lower()
            bbox = getattr(it, "bbox", None)
            # Ensure bbox is tuple if present
            if bbox and hasattr(bbox, "as_tuple"):
                bbox = bbox.as_tuple()
            
            node_id = f"{Path(pdf_path).name}::p{page_idx}::{node_seq}::{sha1(md)[:6]}"
            nodes.append(ElementNode(
                id=node_id, 
                page=page_idx, 
                kind=kind, 
                md=md, 
                bbox=bbox, 
                source_path=str(pdf_path)
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
            node_id = f"{Path(pdf_path).name}::p{page_idx+1}::{node_seq}::{sha1(text)[:6]}"
            nodes.append(ElementNode(
                id=node_id, 
                page=page_idx+1, 
                kind=kind, 
                md=text, 
                bbox=(x0, y0, x1, y1), 
                source_path=str(pdf_path)
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
    chunks: List[Chunk] = []
    by_page: Dict[int, List[ElementNode]] = {}
    for n in nodes:
        by_page.setdefault(n.page, []).append(n)

    for page, arr in by_page.items():
        for i, n in enumerate(arr):
            left = max(0, i - k_neighbors)
            right = min(len(arr), i + k_neighbors + 1)
            neigh = [x for j, x in enumerate(arr[left:right]) if j + left != i]

            # Construct markdown with context annotations
            md_parts = [f"<!-- NODE {n.id} ({n.kind}) -->\n{n.md}"]
            for nb in neigh:
                md_parts.append(f"\n<!-- NEIGHBOR {nb.id} ({nb.kind}) -->\n{nb.md}")

            chunk_id = f"{n.id}::k{k_neighbors}"
            neighbor_chunk_ids = [f"{nb.id}::k{k_neighbors}" for nb in neigh]

            chunks.append(Chunk(
                chunk_id=chunk_id,
                element_id=n.id,
                page=page,
                md="\n".join(md_parts),
                neighbor_ids=neighbor_chunk_ids,
                source_path=n.source_path
            ))
    return chunks


# -----------------------------------------------------------------------------
# Embedding Logic
# -----------------------------------------------------------------------------

_EMBEDDER_CACHE: Dict[str, Any] = {}

def get_vllm_client():
    from openai import OpenAI
    api_base = os.getenv("VLLM_EMB_URL", "http://localhost:8001/v1")
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
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
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch = [t.replace("\n", " ") for t in batch] 
                
                resp = client.embeddings.create(input=batch, model=EMB_MODEL_REPO)
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
            chunk_id TEXT UNIQUE,
            element_id TEXT,
            page INTEGER,
            md TEXT,
            neighbors TEXT,
            source_path TEXT
        );
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files_meta (
            source_path TEXT PRIMARY KEY,
            description TEXT,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def _pack_f32(arr: np.ndarray) -> bytes:
    return arr.astype("float32").tobytes()

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
            INSERT OR REPLACE INTO chunks_meta(chunk_id, element_id, page, md, neighbors, source_path)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (ch.chunk_id, ch.element_id, ch.page, ch.md, json.dumps(ch.neighbor_ids), ch.source_path))
        
        cur.execute("SELECT rowid FROM chunks_meta WHERE chunk_id = ?", (ch.chunk_id,))
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
    Performs semantic search in the DB, returning the Top-K results FOR EACH document.
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
        if source_filter:
            sf = source_filter.strip()
            # Apply source restriction inside vector query to avoid global-topk truncation bias.
            rows = conn.execute(
                """
                SELECT v.rowid, v.distance
                FROM v_chunks v
                JOIN chunks_meta m ON m.rowid = v.rowid
                WHERE v.embedding MATCH ?
                  AND k = ?
                  AND (
                    m.source_path = ?
                    OR m.source_path LIKE ?
                    OR m.source_path LIKE ?
                  )
                ORDER BY v.distance ASC;
                """,
                (q_blob, top_k, sf, f"%/{sf}", f"%\\{sf}"),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT v.rowid, v.distance
                FROM v_chunks v
                WHERE v.embedding MATCH ?
                  AND k = ?
                ORDER BY v.distance ASC;
                """,
                (q_blob, top_k),
            ).fetchall()

        all_results = []
        
        for r in rows:
            row_id, dist = r
            
            # Convert L2 distance to Cosine Similarity
            sim = max(0.0, min(1.0, 1.0 - (dist**2) / 2.0))
            
            meta_row = conn.execute("""
                SELECT chunk_id, element_id, page, md, neighbors, source_path
                FROM chunks_meta
                WHERE rowid = ?
            """, (row_id,)).fetchone()
            
            if meta_row:
                s_path = meta_row[5]
                
                all_results.append({
                    "score": sim,
                    "distance": dist,
                    "chunk_id": meta_row[0],
                    "element_id": meta_row[1],
                    "page": meta_row[2],
                    "md": meta_row[3],
                    "neighbors": json.loads(meta_row[4]) if meta_row[4] else [],
                    "source_path": s_path
                })
                
                if len(all_results) >= top_k:
                    break

        # Sort desc by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results

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
            "dim": EMB_DIM
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
            SELECT chunk_id, element_id, page, source_path, neighbors, md
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
    for chunk_id, element_id, page, source_path, neighbors_raw, md in rows:
        try:
            neighbors = json.loads(neighbors_raw) if neighbors_raw else []
        except Exception:
            neighbors = []
        preview = (md or "").replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:220] + "..."
        out.append(
            {
                "chunk_id": chunk_id,
                "element_id": element_id,
                "page": page,
                "source_path": source_path,
                "neighbors_count": len(neighbors),
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
        texts = [c.md for c in chunks]
        embeddings = embed_texts(texts)
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}

    _update_progress(0.9, "Storing vectors in database...")

    # 4. Store
    try:
        init_db(db_path) 
        upsert_chunks(db_path, chunks, embeddings)
        
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
