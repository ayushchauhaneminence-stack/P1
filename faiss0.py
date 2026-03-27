import os
import json
import time
import numpy as np
import faiss
import requests

from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

# ── File paths ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
FORMATTED_TXT = os.getenv("FORMATTED_TXT", "/home/webexpert/shell/1P_chatbot/formatted_Output.txt")
FAISS_INDEX   = os.path.join(BASE_DIR, "faiss.index")
METADATA_JSON = os.path.join(BASE_DIR, "faiss_metadata.json")

# ── Ollama config ───────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Separator written by groq_chunker.py between every chunk
CHUNK_SEP = "*" * 70
LINE_SEP  = "=" * 70

# ============================================================
#  FastAPI app
# ============================================================

app = FastAPI(title="FAISS Index API (Ollama)", version="2.0.0")

_index: Optional[faiss.Index] = None
_meta:  list = []


@app.on_event("startup")
def _startup():
    try:
        resp   = requests.get(OLLAMA_BASE_URL + "/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        print("[startup] Ollama reachable. Models: " + str(models))
        short = [m.split(":")[0] for m in models]
        if OLLAMA_EMBED_MODEL.split(":")[0] not in short:
            print("[startup] WARNING: '" + OLLAMA_EMBED_MODEL + "' not pulled yet.")
            print("[startup] Run:  ollama pull " + OLLAMA_EMBED_MODEL)
    except Exception as exc:
        print("[startup] WARNING: Cannot reach Ollama — " + str(exc))

    if os.path.exists(FAISS_INDEX) and os.path.exists(METADATA_JSON):
        _load_index()
        print("[startup] Existing index loaded — " + str(len(_meta)) + " chunks.")


# ============================================================
#  Ollama embedding helpers
# ============================================================

def _embed_texts(texts: list, batch_size: int = 32) -> np.ndarray:
    """Call Ollama /api/embed in batches. Returns float32 array (N, dim)."""
    all_embeddings = []
    total = len(texts)
    n_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        print("[embed] Batch " + str(batch_num) + "/" + str(n_batches)
              + "  (" + str(len(batch)) + " texts)")

        resp = requests.post(
            OLLAMA_BASE_URL + "/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": batch},
            timeout=120,
        )

        if not resp.ok:
            raise RuntimeError(
                "Ollama embed failed [" + str(resp.status_code) + "]: " + resp.text
            )

        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError("Unexpected Ollama response: " + str(data))

        all_embeddings.extend(embeddings)

    return np.array(all_embeddings, dtype="float32")


def _embed_single(text: str) -> np.ndarray:
    return _embed_texts([text])


# ============================================================
#  Parser — reads formatted_Output.txt produced by groq_chunker.py
#
#  Expected file structure:
#
#  SEMANTIC CHUNKS - LLM-generated
#  Generated : ...
#  Total     : N chunks
#  ======================================================================
#
#  [ Chunk 1 of N ]
#
#    Optional heading
#
#  text text text ...
#
#  ***...*** (70 asterisks)
#  [ Chunk 2 of N ]
#  ...
# ============================================================

def _parse_formatted_txt(path: str) -> list:
    """
    Parse formatted_Output.txt written by groq_chunker.py.

    Strategy (zero regex):
    1. Read the whole file as one string.
    2. Split on CHUNK_SEP  ("*" * 70)  — each piece is one raw chunk block.
    3. For each block, split on newlines and walk the lines to extract:
         - chunk label   (line that starts with "[ Chunk ")
         - heading       (indented line right after the label, if present)
         - body text     (everything else that is non-empty)
    4. Skip the file header (lines before the first chunk label).
    Returns a list of dicts ready for embedding.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("formatted_Output.txt not found at: " + path)

    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()

    # Split file into raw blocks on the chunk separator
    raw_blocks = raw.split(CHUNK_SEP)

    chunks = []
    chunk_index = 0

    for block in raw_blocks:
        lines = block.splitlines()

        # Walk lines to find chunk label
        label_found = False
        heading     = ""
        body_lines  = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Detect chunk label  e.g.  [ Chunk 3 of 17 ]
            if stripped.startswith("[ Chunk ") and stripped.endswith("]"):
                label_found = True
                i += 1
                continue

            if not label_found:
                i += 1
                continue

            # Skip the file-header separator line (= * 70 / = * 60 etc.)
            if stripped and all(c in ("=", "-") for c in stripped):
                i += 1
                continue

            # First non-empty, indented (2+ spaces) line after label = heading
            if (
                not heading
                and stripped
                and len(line) > len(stripped)    # has leading whitespace
                and len(stripped) <= 120
            ):
                heading = stripped
                i += 1
                continue

            # Everything else is body
            body_lines.append(stripped)
            i += 1

        if not label_found:
            continue

        body = " ".join(ln for ln in body_lines if ln).strip()

        if not body:
            continue

        chunks.append({
            "chunk_index": chunk_index,
            "heading":     heading,
            "text":        body,
            "word_count":  len(body.split()),
        })
        chunk_index += 1

    return chunks


# ============================================================
#  Build / save / load FAISS index
# ============================================================

def _build_index(chunks: list) -> faiss.Index:
    texts = [
        (c["heading"] + ". " + c["text"]) if c["heading"] else c["text"]
        for c in chunks
    ]

    print("[index] Embedding " + str(len(texts)) + " chunks via Ollama ("
          + OLLAMA_EMBED_MODEL + ")...")
    embeddings = _embed_texts(texts)

    faiss.normalize_L2(embeddings)   # cosine similarity via inner product

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("[index] Built — " + str(index.ntotal) + " vectors, dim=" + str(dim))
    return index


def _save_index(index: faiss.Index, meta: list):
    faiss.write_index(index, FAISS_INDEX)
    with open(METADATA_JSON, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def _load_index():
    global _index, _meta
    _index = faiss.read_index(FAISS_INDEX)
    with open(METADATA_JSON, "r", encoding="utf-8") as fh:
        _meta = json.load(fh)


# ============================================================
#  Schemas
# ============================================================

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# ============================================================
#  Endpoints
# ============================================================

@app.post("/index/build",
          summary="Parse formatted_Output.txt → embed via Ollama → build FAISS index")
async def build(
    txt_path: Optional[str] = Query(None, description="Override path to formatted_Output.txt"),
):
    global _index, _meta

    src = txt_path or FORMATTED_TXT

    # ── 1. Parse ─────────────────────────────────────────────────────────
    try:
        chunks = _parse_formatted_txt(src)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                "No chunks found in '" + src + "'. "
                "Make sure you ran POST /format (groq_chunker.py) first "
                "and the file contains '[ Chunk N of M ]' markers."
            ),
        )

    print("[build] Parsed " + str(len(chunks)) + " chunks from " + src)

    # ── 2. Embed + index ─────────────────────────────────────────────────
    try:
        _meta  = chunks
        _index = _build_index(chunks)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail="Ollama error: " + str(exc))

    # ── 3. Persist ────────────────────────────────────────────────────────
    _save_index(_index, _meta)

    return {
        "status":        "ok",
        "source":        src,
        "total_chunks":  len(chunks),
        "embed_model":   OLLAMA_EMBED_MODEL,
        "index_file":    FAISS_INDEX,
        "metadata_file": METADATA_JSON,
        "embedding_dim": _index.d,
        "sample_chunks": [
            {"chunk_index": c["chunk_index"],
             "heading":     c["heading"],
             "preview":     c["text"][:120] + ("..." if len(c["text"]) > 120 else "")}
            for c in chunks[:3]
        ],
    }


@app.post("/index/query", summary="Semantic search over the FAISS index")
async def query(body: QueryRequest):
    if _index is None or not _meta:
        raise HTTPException(
            status_code=404,
            detail="Index not built yet. Call POST /index/build first.",
        )

    try:
        q_vec = _embed_single(body.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail="Ollama error: " + str(exc))

    faiss.normalize_L2(q_vec)

    fetch_k         = min(body.top_k, len(_meta))
    scores, indices = _index.search(q_vec, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = _meta[idx]
        results.append({
            "score":       round(float(score), 4),
            "chunk_index": chunk["chunk_index"],
            "heading":     chunk["heading"],
            "text":        chunk["text"],
            "word_count":  chunk["word_count"],
        })

    return {"query": body.query, "results": results}


@app.post("/index/debug-parse",
          summary="Preview what the parser extracts (without building the index)")
async def debug_parse(
    txt_path: Optional[str] = Query(None, description="Path to formatted_Output.txt"),
):
    """
    Useful for diagnosing 422 errors.
    Returns the first 5 parsed chunks and the total count.
    """
    src = txt_path or FORMATTED_TXT
    try:
        chunks = _parse_formatted_txt(src)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "source":       src,
        "total_chunks": len(chunks),
        "preview": [
            {
                "chunk_index": c["chunk_index"],
                "heading":     c["heading"],
                "word_count":  c["word_count"],
                "text_preview": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
            }
            for c in chunks[:5]
        ],
    }


@app.get("/index/info", summary="Info about the currently loaded FAISS index")
async def index_info():
    if _index is None:
        return {"status": "no index loaded"}

    return {
        "status":        "loaded",
        "embed_model":   OLLAMA_EMBED_MODEL,
        "total_vectors": _index.ntotal,
        "embedding_dim": _index.d,
        "total_chunks":  len(_meta),
        "index_file":    FAISS_INDEX,
    }


@app.get("/index/chunks", summary="Browse all indexed chunks")
async def get_chunks():
    if not _meta:
        raise HTTPException(status_code=404, detail="No index loaded. Call POST /index/build first.")
    return {"total": len(_meta), "chunks": _meta}


@app.get("/ollama/models", summary="List models available in local Ollama")
async def ollama_models():
    try:
        resp = requests.get(OLLAMA_BASE_URL + "/api/tags", timeout=5)
        return resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Cannot reach Ollama: " + str(exc))