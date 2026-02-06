from __future__ import annotations

"""
Catálogo de herramientas (tools) para el Agnostic Deep Agent 2026.

Incluye:
- Tools "toy" (to_upper, word_count, is_palindrome).
- Tools matemáticas seguras (eval_math_expression, sum_numbers, average_numbers).
- Tools de modelos Qwen3 locales:
    - embed_texts              → Qwen3-Embedding (núcleo reutilizable).
    - semantic_search          → búsqueda semántica sobre una lista de textos.
    - semantic_search_in_csv   → búsqueda semántica sobre filas de un CSV.
    - embed_context_tables     → precálculo de embeddings para tablas de contexto.
    - judge_row_with_context   → juicio simple de una fila usando contexto tabular.
    - rerank_qwen3             → Qwen3-Reranker

El registro global TOOL_REGISTRY permite seleccionar tools por nombre
y construir subconjuntos según la configuración (p.ej. setup.yaml).
"""

from typing import List, Dict, Any

from langchain_core.tools import tool

import os
import ast
import operator as _op
import numbers

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,              # Qwen3-Embedding
    AutoModelForCausalLM,   # Qwen3-Reranker (via logits yes/no)
)

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# TOOLS "toy"
# ─────────────────────────────────────────────

@tool
def to_upper(text: str) -> str:
    """Convierte el texto a MAYÚSCULAS."""
    return text.upper()


@tool
def word_count(text: str) -> int:
    """Devuelve el número de palabras en el texto."""
    return len([w for w in text.split() if w])


@tool
def is_palindrome(text: str) -> bool:
    """True si el texto (sin espacios/casos) es palíndromo."""
    s = "".join(ch.lower() for ch in text if ch.isalnum())
    return s == s[::-1]


# ─────────────────────────────────────────────
# TOOLS matemáticas (evaluadas en Python)
# ─────────────────────────────────────────────

# Operadores permitidos en la expresión matemática
_ALLOWED_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.Mod: _op.mod,
}


def _eval_ast(node: ast.AST) -> float:
    """Evalúa de forma segura un AST restringido a operaciones aritméticas básicas."""
    if isinstance(node, ast.Num):  # Python <3.8
        return node.n
    if isinstance(node, ast.Constant):  # números en 3.8+
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Sólo se permiten números en las constantes.")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Operador no permitido: {op_type.__name__}")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _ALLOWED_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand)

    raise ValueError(f"Nodo de AST no permitido: {type(node).__name__}")


@tool
def eval_math_expression(expression: str) -> float:
    """
    Evalúa una expresión matemática sencilla usando Python de forma segura.

    Soporta:
      - suma, resta, multiplicación, división, módulo, potencias
      - paréntesis
      - signos unarios (p.ej. -3)

    Ejemplos válidos:
      "1 + 2 * 3"
      "(10 - 4) / 2"
      "2**3 + 5"

    NOTA:
      El operador de potencia soportado es **, NO ^ (que en Python es XOR).
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_ast(parsed.body)
        return float(result)
    except Exception as exc:
        raise ValueError(
            f"No se pudo evaluar la expresión: {expression!r}. Error: {exc}"
        ) from exc


# ─────────────────────────────────────────────
# Helpers numéricos robustos para sum/avg
# ─────────────────────────────────────────────

def _coerce_to_float(x: Any) -> float:
    """
    Intenta convertir un valor genérico a float.

    Soporta:
    - ints/floats/np.number
    - strings numéricos ("3.14")
    - dicts con claves típicas: "value", "val", "number", "num"
    """
    if isinstance(x, numbers.Number):
        return float(x)

    if isinstance(x, str):
        # Permite strings como "3.14", "42"
        return float(x.strip())

    if isinstance(x, dict):
        for key in ("value", "val", "number", "num"):
            if key in x:
                return _coerce_to_float(x[key])

    raise ValueError(f"No se pudo interpretar {x!r} como número.")


@tool
def sum_numbers(numbers: List[Any]) -> float:
    """
    Devuelve la suma de una lista de números.

    La tool es robusta: acepta tanto números puros como estructuras
    que contengan el número, por ejemplo:

      - [1, 2.5, "3"]
      - [{"value": 10}, {"number": "20"}]

    Para evitar errores de validación Pydantic, el tipo es List[Any]
    y se hace coerción interna a float.
    """
    if not isinstance(numbers, list):
        raise ValueError("El parámetro 'numbers' debe ser una lista.")

    vals = [_coerce_to_float(n) for n in numbers]
    return float(sum(vals))


@tool
def average_numbers(numbers: List[Any]) -> float:
    """
    Devuelve la media aritmética de una lista de números.

    Mismo comportamiento robusto que sum_numbers:
      - [1, 2.5, "3"]
      - [{"value": 10}, {"number": "20"}]
    """
    if not isinstance(numbers, list):
        raise ValueError("El parámetro 'numbers' debe ser una lista.")
    if not numbers:
        raise ValueError("La lista de números está vacía.")

    vals = [_coerce_to_float(n) for n in numbers]
    return float(sum(vals) / len(vals))


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

    Se usa internamente por:
      - embed_texts (tool)
      - semantic_search
      - semantic_search_in_csv
      - embed_context_tables
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


@tool
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Devuelve embeddings Qwen3-Embedding para cada texto, usando Transformers local.

    - Usa EMB_MODEL_ID (por defecto Qwen/Qwen3-Embedding-0.6B).
    - Por defecto corre en CPU (QWEN_EMB_USE_CUDA=0).
    - Devuelve una lista de vectores (list[list[float]]).

    NOTA:
    - Internamente usa _embed_texts_core para compartir el mismo estado
      con otras tools (semantic_search, semantic_search_in_csv, embed_context_tables).
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
    Búsqueda semántica simple usando Qwen3-Embedding sobre una lista de textos.

    Pensado para casos como:
      - Dado un query, rankear párrafos, definiciones, cláusulas, etc.
      - Integrar tablas ya "serializadas" a textos (fila → string).

    Parámetros:
      - query: texto de búsqueda.
      - documents: lista de textos candidatos.
      - top_k: número máximo de resultados a devolver.

    Devuelve una lista de dicts:
      [{ "index": int, "document": str, "score": float }, ...]
    ordenada por score descendente.
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

    - csv_path: ruta al CSV (p.ej. parametrias.csv, diccionario_abreviaturas.csv).
    - text_columns: columnas a concatenar para generar el texto base.

    Devuelve un dict:
      {
        "df": DataFrame,
        "emb": np.ndarray (n_filas, d),
        "texts": List[str]  # representación textual de cada fila
      }
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


@tool
def semantic_search_in_csv(
    query: str,
    csv_path: str,
    text_columns: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Búsqueda semántica sobre filas de un CSV usando Qwen3-Embedding.

    Casos típicos:
      - csv_path="parametrias.csv", text_columns=["campo", "descripcion", "regla"]
      - csv_path="diccionario_abreviaturas.csv", text_columns=["abreviatura", "definicion"]

    Parámetros:
      - query: texto de búsqueda (ej. "monto máximo de crédito hipotecario").
      - csv_path: ruta al archivo CSV.
      - text_columns: columnas que se concatenan para representar cada fila.
      - top_k: número máximo de filas a devolver.

    Devuelve:
      [
        {
          "row_index": int,
          "score": float,
          "row": {col: valor, ...},
          "text": "col1: ... | col2: ..."
        },
        ...
      ]
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
    Precálcula embeddings por fila para varias tablas de contexto (CSV):

      - Parametrías (reglas, umbrales, categorías, etc.).
      - Diccionarios de abreviaturas/definiciones.
      - Otras tablas de contexto que quieras.

    Parámetros:
      - table_paths: lista de rutas a CSVs de contexto.
      - text_columns: mapa opcional {path: [cols,...]} con las columnas
        que se usarán para generar los textos por fila. Si no se indica
        para un path, se usan todas las columnas del CSV.

    Efectos:
      - Llama internamente a _get_csv_embeddings(...) para cada tabla,
        poblando la caché interna _CSV_EMB_CACHE.
      - Reutiliza el mismo modelo de embeddings que embed_texts.

    Devuelve un resumen:

      {
        "tables": [
          {
            "path": "<ruta_csv>",
            "n_rows": int,
            "n_cols": int,
            "text_columns": [ ... ]
          },
          ...
        ],
        "embedding_dim": int | null
      }
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
                # Si hay discrepancia, mantenemos el primero y no reventamos
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


@tool
def rerank_qwen3(query: str, documents: List[str]) -> List[Dict[str, Any]]:
    """
    Usa Qwen3-Reranker (local, vía Transformers) para ordenar documentos por relevancia.

    - Usa RERANK_MODEL_ID (por defecto Qwen/Qwen3-Reranker-0.6B).
    - Por defecto corre en CPU (QWEN_RERANK_USE_CUDA=0).
    - Devuelve [{index, document, score}] ordenado por score desc.
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

    Pensado para el caso:
      - row: fila de atributos de un contrato (input A).
      - param_hits: resultados de semantic_search_in_csv sobre parametrías.
      - glossary_hits: resultados de semantic_search_in_csv sobre diccionario
        de abreviaturas / definiciones.

    NO usa modelos de lenguaje dentro de la tool; es puramente determinista
    y genérico. El LLM puede decidir cómo interpretar el resultado.

    Reglas sencillas:
      - Si NO hay hits ni en param_hits ni en glossary_hits → "review_required".
      - Si hay al menos un hit en alguna de las dos → "ok".
      - Se devuelve también un pequeño resumen de motivos.

    Devuelve un dict como:

      {
        "contract_id": str | int | null,
        "judgement": "ok" | "review_required",
        "reasons": [str, ...],
        "row": row,
        "param_hits": [...],
        "glossary_hits": [...],
      }
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
# Registro de tools
# ─────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, Any] = {
    "to_upper": to_upper,
    "word_count": word_count,
    "is_palindrome": is_palindrome,
    "eval_math_expression": eval_math_expression,
    "sum_numbers": sum_numbers,
    "average_numbers": average_numbers,
    "embed_texts": embed_texts,
    "semantic_search": semantic_search,
    "semantic_search_in_csv": semantic_search_in_csv,
    "embed_context_tables": embed_context_tables,
    "rerank_qwen3": rerank_qwen3,
    "judge_row_with_context": judge_row_with_context,
}


def get_default_tools(enabled_names: List[str] | None = None) -> List[Any]:
    """
    Tools por defecto del agente agnóstico.

    - Si enabled_names es None → devuelve TODAS las tools registradas.
    - Si enabled_names es una lista → sólo devuelve las que estén en TOOL_REGISTRY.
    """
    if enabled_names is None:
        return list(TOOL_REGISTRY.values())
    return [TOOL_REGISTRY[name] for name in enabled_names if name in TOOL_REGISTRY]


def get_tools_by_names(names: List[str]) -> List[Any]:
    """
    Permite seleccionar tools por nombre desde TOOL_REGISTRY.
    Equivalente a get_default_tools(names), se mantiene por claridad semántica.
    """
    return get_default_tools(names)
