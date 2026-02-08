from typing import List, Dict, Any
import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from agnostic_agent.tools.decorators import tool

# ─────────────────────────────────────────────
# QWEN3-EMBEDDING – Transformers local
# ─────────────────────────────────────────────

_EMB_STATE: Dict[str, Any] = {}


def _ensure_embedding_loaded() -> None:
    """Carga Qwen3-Embedding una sola vez en memoria."""
    global _EMB_STATE
    if _EMB_STATE:
        return

    model_id = os.getenv("EMB_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")

    # Permite forzar device vía env si quieres:
    #   QWEN_EMB_DEVICE = "cuda" | "cpu"
    forced_device = os.getenv("QWEN_EMB_DEVICE")
    if forced_device in ("cuda", "cpu"):
        device = forced_device
    else:
        use_cuda = (
            os.getenv("QWEN_EMB_USE_CUDA", "0").lower() in ("1", "true", "yes")
            and torch.cuda.is_available()
        )
        device = "cuda" if use_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    model.eval()

    max_length = int(os.getenv("QWEN_EMB_MAX_LEN", "512"))

    _EMB_STATE.update(
        {
            "model_id": model_id,
            "device": device,
            "tokenizer": tokenizer,
            "model": model,
            "max_length": max_length,
        }
    )


def _embed_texts_core(inputs: List[str]) -> np.ndarray:
    """
    Núcleo de embeddings: recibe una lista de textos y devuelve un array (n, d).
    """
    _ensure_embedding_loaded()
    state = _EMB_STATE

    tokenizer = state["tokenizer"]
    model = state["model"]
    device = state["device"]
    max_length = state["max_length"]

    if not inputs:
        return np.zeros((0, 0), dtype="float32")

    enc = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        emb = last_hidden.mean(dim=1)           # (batch, hidden)

    return emb.cpu().numpy()


@tool(mode="public", output_schema={"type": "array", "items": {"type": "array", "items": {"type": "number"}}})
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Devuelve embeddings Qwen3-Embedding para cada texto, usando Transformers local.
    """
    if isinstance(texts, str):
        inputs = [texts]
    else:
        inputs = list(texts)

    if not inputs:
        return []

    emb = _embed_texts_core(inputs)
    return emb.tolist()


# ─────────────────────────────────────────────
# BÚSQUEDA SEMÁNTICA GENÉRICA (en memoria)
# ─────────────────────────────────────────────

def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Matriz de similitud coseno entre:
      - a: (n, d)
      - b: (m, d)

    Devuelve: (n, m).
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype="float32")

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.matmul(a_norm, b_norm.T)


@tool
def semantic_search(
    query: str,
    documents: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Búsqueda semántica simple sobre una lista de textos EN MEMORIA.
    
    IMPORTANTE:
    - Requiere 'documents' como lista de strings explícita.
    - NO USAR para archivos CSV o rutas de archivo. Para CSV usa 'semantic_search_in_csv'.
    - Úsalo solo si ya tienes textos en memoria que quieres reordenar/filtrar.
    """
    if isinstance(documents, str):
        docs = [documents]
    else:
        docs = list(documents)

    if not docs:
        return []

    # Embeddings: query (1, d) y docs (n, d)
    query_emb = _embed_texts_core([query])          # (1, d)
    docs_emb = _embed_texts_core(docs)              # (n, d)

    sims = _cosine_sim_matrix(query_emb, docs_emb)  # (1, n)
    scores = sims[0]

    # Top-k
    top_k = max(1, min(top_k, len(docs)))
    indices = np.argsort(-scores)[:top_k]

    results: List[Dict[str, Any]] = []
    for idx in indices:
        results.append(
            {
                "index": int(idx),
                "document": docs[idx],
                "score": float(scores[idx]),
            }
        )

    return results


# ─────────────────────────────────────────────
# BÚSQUEDA SEMÁNTICA EN CSV (parametrías, diccionarios, etc.)
# ─────────────────────────────────────────────

# Cache por (csv_path, columnas_join) → { "df": DataFrame, "emb": np.ndarray, "texts": List[str] }
_CSV_EMB_CACHE: Dict[str, Any] = {}


def _get_csv_embeddings(
    csv_path: str,
    text_columns: List[str],
) -> Dict[str, Any]:
    """
    Carga (o reutiliza) embeddings por fila para un CSV.
    """
    key = f"{os.path.abspath(csv_path)}|{'|'.join(text_columns)}"
    if key in _CSV_EMB_CACHE:
        return _CSV_EMB_CACHE[key]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalizamos text_columns a las que existan
    cols = [c for c in text_columns if c in df.columns]
    if not cols:
        raise ValueError(
            f"Ninguna de las columnas {text_columns!r} existe en el CSV {csv_path!r}."
        )

    texts: List[str] = []
    for _, row in df.iterrows():
        parts = []
        for col in cols:
            val = row.get(col, "")
            if pd.isna(val):
                continue
            parts.append(f"{col}: {val}")
        texts.append(" | ".join(parts))

    emb = _embed_texts_core(texts)  # (n_filas, d)

    payload = {
        "df": df,
        "emb": emb,
        "texts": texts,
    }
    _CSV_EMB_CACHE[key] = payload
    return payload


@tool(mode="public", output_schema={"type": "array", "items": {"type": "object"}})
def semantic_search_in_csv(
    query: str,
    csv_path: str,
    text_columns: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Búsqueda semántica sobre filas de un CSV usando Qwen3-Embedding.
    """
    payload = _get_csv_embeddings(csv_path, text_columns)
    df: pd.DataFrame = payload["df"]
    emb: np.ndarray = payload["emb"]

    if df.empty:
        return []

    # Embedding del query
    query_emb = _embed_texts_core([query])  # (1, d)
    sims = _cosine_sim_matrix(query_emb, emb)[0]  # (n_filas,)

    top_k = max(1, min(top_k, len(df)))
    indices = np.argsort(-sims)[:top_k]

    results: List[Dict[str, Any]] = []
    for idx in indices:
        row_data = df.iloc[int(idx)].to_dict()
        results.append(
            {
                "row_index": int(idx),
                "score": float(sims[idx]),
                "row": row_data,
                "text": payload["texts"][int(idx)],
            }
        )

    return results


# ─────────────────────────────────────────────
# CONTEXTO: precálculo de embeddings de tablas
# ─────────────────────────────────────────────

@tool
def embed_context_tables(
    table_paths: List[str],
    text_columns: Dict[str, List[str]] | None = None,
) -> Dict[str, Any]:
    """
    Precálcula embeddings por fila para varias tablas de contexto (CSV).
    """
    if isinstance(table_paths, str):
        paths = [table_paths]
    else:
        paths = list(table_paths)

    tables_info: List[Dict[str, Any]] = []
    emb_dim = None

    for p in paths:
        # Determinar columnas de texto para este path
        cols = None
        if text_columns and isinstance(text_columns, dict):
            cols = text_columns.get(p)

        if not cols:
            # Si no se especifican columnas, usamos todas
            if not os.path.exists(p):
                raise FileNotFoundError(f"No se encontró el CSV: {p}")
            df_head = pd.read_csv(p, nrows=1)
            cols = list(df_head.columns)

        payload = _get_csv_embeddings(p, cols)
        df = payload["df"]
        emb = payload["emb"]

        if emb.size > 0:
            d = emb.shape[1]
            if emb_dim is None:
                emb_dim = d
            elif emb_dim != d:
                pass

        tables_info.append(
            {
                "path": p,
                "n_rows": int(len(df)),
                "n_cols": int(len(df.columns)),
                "text_columns": cols,
            }
        )

    return {
        "tables": tables_info,
        "embedding_dim": emb_dim,
    }


# ─────────────────────────────────────────────
# QWEN3-RERANKER – Transformers local
# ─────────────────────────────────────────────

_RERANK_STATE: Dict[str, Any] = {}


def _ensure_reranker_loaded() -> None:
    """Carga Qwen3-Reranker una sola vez en memoria."""
    global _RERANK_STATE
    if _RERANK_STATE:
        return

    model_id = os.getenv("RERANK_MODEL_ID", "Qwen/Qwen3-Reranker-0.6B")

    forced_device = os.getenv("QWEN_RERANK_DEVICE")
    if forced_device in ("cuda", "cpu"):
        device = forced_device
    else:
        use_cuda = (
            os.getenv("QWEN_RERANK_USE_CUDA", "0").lower() in ("1", "true", "yes")
            and torch.cuda.is_available()
        )
        device = "cuda" if use_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Importante: trust_remote_code=True para Qwen3-Reranker
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    true_token_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token_id = tokenizer("no", add_special_tokens=False).input_ids[0]

    max_length = int(os.getenv("QWEN_RERANK_MAX_LEN", "1024"))

    _RERANK_STATE.update(
        {
            "model_id": model_id,
            "device": device,
            "tokenizer": tokenizer,
            "model": model,
            "true_token_id": true_token_id,
            "false_token_id": false_token_id,
            "max_length": max_length,
        }
    )


def _format_rerank_prompts(
    query: str,
    docs: List[str],
    instruction: str,
) -> List[str]:
    prompts: List[str] = []
    for doc in docs:
        text = (
            "You are a relevance judge. "
            "Decide if the document answers the query.\n\n"
            f"Instruction: {instruction}\n"
            f"Query: {query}\n"
            f"Document: {doc}\n\n"
            "Answer with 'yes' if it is relevant, otherwise 'no'."
        )
        prompts.append(text)
    return prompts


@tool(mode="public", output_schema={"type": "array", "items": {"type": "object"}})
def rerank_qwen3(query: str, documents: List[str]) -> List[Dict[str, Any]]:
    """
    Usa Qwen3-Reranker (local, vía Transformers) para ordenar documentos por relevancia.
    """
    _ensure_reranker_loaded()
    state = _RERANK_STATE

    tokenizer = state["tokenizer"]
    model = state["model"]
    device = state["device"]
    true_token_id = state["true_token_id"]
    false_token_id = state["false_token_id"]
    max_length = state["max_length"]

    if isinstance(documents, str):
        docs = [documents]
    else:
        docs = list(documents)

    if not docs:
        return []

    instruction = os.getenv(
        "QWEN_RERANK_INSTRUCT",
        "Given a web search query, rank documents by how well they answer the query.",
    )

    prompts = _format_rerank_prompts(query, docs, instruction)

    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits[:, -1, :]   # (batch, vocab)
        yes_logits = logits[:, true_token_id]
        no_logits = logits[:, false_token_id]
        stacked = torch.stack([no_logits, yes_logits], dim=-1)  # (batch, 2)
        probs = torch.nn.functional.softmax(stacked, dim=-1)[:, 1].tolist()

    results: List[Dict[str, Any]] = []
    for idx, (doc, score) in enumerate(zip(docs, probs)):
        results.append(
            {
                "index": idx,
                "document": doc,
                "score": float(score),
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ─────────────────────────────────────────────
# JUICIO FILA + CONTEXTO (parametrías / diccionarios)
# ─────────────────────────────────────────────

@tool
def judge_row_with_context(
    row: Dict[str, Any],
    param_hits: List[Dict[str, Any]] | None = None,
    glossary_hits: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Aplica un juicio simple sobre una fila de atributos usando hits de contexto.
    """
    param_hits = param_hits or []
    glossary_hits = glossary_hits or []

    # Heurística para encontrar un identificador de contrato/fila
    contract_id = (
        row.get("numero_contrato")
        or row.get("numero de contrato")
        or row.get("num_contrato")
        or row.get("contract_number")
        or row.get("contract_id")
        or row.get("id")
    )

    has_context = bool(param_hits or glossary_hits)
    judgement = "ok" if has_context else "review_required"

    reasons: List[str] = []
    if not has_context:
        reasons.append(
            "No se encontraron coincidencias ni en las parametrías ni en el diccionario; "
            "se recomienda revisión manual."
        )
    else:
        if param_hits:
            best_param = param_hits[0]
            desc = best_param.get("text") or str(best_param.get("row", ""))[:200]
            reasons.append(
                f"Se encontraron al menos {len(param_hits)} filas relevantes en la tabla de parametrías. "
                f"Ejemplo: {desc}"
            )
        if glossary_hits:
            best_gl = glossary_hits[0]
            desc = best_gl.get("text") or str(best_gl.get("row", ""))[:200]
            reasons.append(
                f"Se encontraron al menos {len(glossary_hits)} filas relevantes en el diccionario. "
                f"Ejemplo: {desc}"
            )

    return {
        "contract_id": contract_id,
        "judgement": judgement,
        "reasons": reasons,
        "row": row,
        "param_hits": param_hits,
        "glossary_hits": glossary_hits,
    }

# ─────────────────────────────────────────────
# KNOWLEDGE BASE SEARCH (Offline / Local DB)
# ─────────────────────────────────────────────

@tool(mode="public", output_schema={"type": "array", "items": {"type": "object"}})
def search_knowledge_base(query: str) -> List[Dict[str, Any]]:
    """
    Primary tool for finding information about specific projects, documents, history, definitions, or any data not in your general training.
    
    Use this tool whenever the user asks about:
    - "El proyecto..." (The project...)
    - "El documento..." (The document...)
    - Specific facts, metrics, or methods described in uploaded files.
    - "Búsqueda en base de conocimiento" (Knowledge Base Search).
    
    Returns relevant text chunks with source metadata.
    """
    try:
        from agnostic_agent.knowledge.vector import search_db
    except ImportError:
        # Fallback if already migrated or missing
        try:
             # Old location fallback
             from agnostic_agent.knowledge_offline import search_db
        except ImportError:
             return [{"error": "Module knowledge not found."}]

    # Default path used in streamlit_app.py
    # Ideally should be in a config/env, but this aligns with current implementation
    db_path = os.path.join(os.getcwd(), "embeddings.db")
    
    if not os.path.exists(db_path):
        return [{"warning": "No knowledge base found (embeddings.db). Please ingest documents via the Offline Manager tab."}]
        
    try:
        results = search_db(db_path, query, top_k=5)
        return results
    except Exception as e:
        return [{"error": f"Search failed: {e}"}]
