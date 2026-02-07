from __future__ import annotations

import os
import json
import hashlib
import sqlite3
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
import torch
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMB_MODEL_REPO = "Qwen/Qwen3-Embedding-0.6B"
# We'll use a local cache dir for models if needed, or rely on HF cache
# For now, let's assume standard HF behavior or a specific dir if set in env
MODELS_CACHE_DIR = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
EMB_DIM = 1024

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

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
    
    # Depending on docling version, iteration might be different. 
    # Attempting a generic approach based on the notebook code.
    # Note: doc.pages might be a dict or list depending on version.
    
    # If pages is a dict-like or we need to iterate differently:
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
                    md = it.export_to_markdown() # Notebook had doc arg, but latest docling might not need it or handles it differently
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
        
        # Check if blocks is empty or None
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
                # We append neighbors for context but main node is first
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
    # Default vLLM embedding port in agnostic setup is often 8001
    api_base = os.getenv("VLLM_EMB_URL", "http://localhost:8001/v1")
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    return OpenAI(base_url=api_base, api_key=api_key)

def check_vllm_embedding_available() -> bool:
    """Checks if vLLM embedding endpoint is responsive."""
    try:
        client = get_vllm_client()
        # Try a dummy embedding
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Try mps
    if device == "cpu":
        try:
             if torch.backends.mps.is_available():
                device = "mps"
        except:
            pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_REPO, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            EMB_MODEL_REPO, 
            trust_remote_code=True, 
            device_map=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32
        )
    except Exception as e:
        logger.error(f"Failed to load model from HF {EMB_MODEL_REPO}: {e}")
        # Fallback to local if needed, but for now re-raise or handle
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

    # 1. Try vLLM first IF configured and available
    # User logs show 404 for embeddings, so vLLM is likely LLM-only.
    # We'll check the env var but default to False if not explicitly set to force usage.
    use_vllm = os.getenv("USE_VLLM_EMBEDDING", "0") == "1"
    
    if use_vllm:
        try:
            client = get_vllm_client()
            # OpenAI API handles batching, but we can respect batch_size too
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
    
    # 2. Local Fallback (CPU/GPU)
    # This is the primary path if vLLM embeddings are disabled (start_emb_server=False)
    tokenizer, model = get_embedder()
    all_vecs = []
    
    # Ensure model is in eval mode
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

    # 1. Float vector table (Virtual Table using vec0)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS v_chunks
        USING vec0(embedding FLOAT[{EMB_DIM}]);
    """)

    # 2. Metadata table
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
        
        # Insert metadata (ignore if exists, or replace? unique chunk_id)
        # Using INSERT OR REPLACE to update content if re-ingesting
        cur.execute("""
            INSERT OR REPLACE INTO chunks_meta(chunk_id, element_id, page, md, neighbors, source_path)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (ch.chunk_id, ch.element_id, ch.page, ch.md, json.dumps(ch.neighbor_ids), ch.source_path))
        
        # Get the rowid (autoincrement from primary key)
        cur.execute("SELECT rowid FROM chunks_meta WHERE chunk_id = ?", (ch.chunk_id,))
        row = cur.fetchone()
        if row:
            row_id = row[0]
            # Upsert vector
            cur.execute("""
                INSERT OR REPLACE INTO v_chunks(rowid, embedding)
                VALUES (?, ?);
            """, (row_id, blob))
            
    conn.commit()
    conn.close()

def search_db(db_path: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Performs semantic search in the DB."""
    if not os.path.exists(db_path):
        return []

    # Embed query
    q_vec = embed_texts([query])[0]
    q_blob = _pack_f32(q_vec)

    import sqlite_vec
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Search
    # sqlite-vec syntax: WHERE embedding MATCH ? AND k = ?
    try:
        rows = conn.execute("""
            SELECT rowid, distance
            FROM v_chunks
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance ASC;
        """, (q_blob, top_k)).fetchall()
    except Exception as e:
        logger.error(f"Search failed: {e}")
        conn.close()
        return []

    if not rows:
        conn.close()
        return []

    # Fetch metadata
    results = []
    for r in rows:
        row_id, dist = r
        meta_row = conn.execute("""
            SELECT chunk_id, element_id, page, md, neighbors, source_path
            FROM chunks_meta
            WHERE rowid = ?
        """, (row_id,)).fetchone()
        
        if meta_row:
            results.append({
                "score": dist, # cosine distance usually? lower is better depending on metric, vec0 default usually L2 or Cosine? 
                               # generic 'distance'. 
                "chunk_id": meta_row[0],
                "element_id": meta_row[1],
                "page": meta_row[2],
                "md": meta_row[3],
                "neighbors": json.loads(meta_row[4]) if meta_row[4] else [],
                "source_path": meta_row[5]
            })
            
    conn.close()
    return results

def get_stats(db_path: str) -> Dict[str, Any]:
    if not os.path.exists(db_path):
        return {"chunks": 0, "files": 0}
        
    conn = sqlite3.connect(db_path)
    try:
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
        n_files = conn.execute("SELECT COUNT(DISTINCT source_path) FROM chunks_meta").fetchone()[0]
        return {"chunks": n_chunks, "files": n_files}
    except:
        return {"chunks": 0, "files": 0}
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# Main Facade
# -----------------------------------------------------------------------------

def ingest_pdf_file(pdf_path: str, db_path: str, k_neighbors: int = 1) -> Dict[str, Any]:
    """High-level function to ingest a PDF."""
    if not os.path.exists(pdf_path):
        return {"error": f"File not found: {pdf_path}"}

    # 1. Parse
    nodes, total_pages = parse_pdf(pdf_path)
    if not nodes:
        return {"error": "No text extracted from PDF."}

    # 2. Chunk
    chunks = build_chunks(nodes, k_neighbors=k_neighbors)
    if not chunks:
        return {"error": "No chunks created."}

    # 3. Embed
    try:
        texts = [c.md for c in chunks]
        embeddings = embed_texts(texts)
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}

    # 4. Store
    try:
        init_db(db_path) # Ensure DB exists
        upsert_chunks(db_path, chunks, embeddings)
    except Exception as e:
        return {"error": f"Database insertion failed: {e}"}

    return {
        "success": True,
        "pages": total_pages,
        "nodes": len(nodes),
        "chunks": len(chunks),
        "file": Path(pdf_path).name
    }
