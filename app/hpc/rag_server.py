#!/usr/bin/env python3
"""
RAG API for FinSight.
- /ingest_dir?path=... : parse & embed all files in a directory (PDF/DOCX/HTML)
- /query {query, k, topn} : retrieve, rerank, ask LLM (via vLLM OpenAI-compatible server),
  and return JSON with reasoning + answer + citations.

This is intentionally simple for day-1:
- Embeddings: BAAI/bge-large-en-v1.5  (dim=1024)  [alternative: intfloat/e5-large-v2]
- Vector store: FAISS (in-memory; saves index + metadata to disk on SIGTERM)
- Reranker: BAAI/bge-reranker-v2-m3  (cross-encoder scorer)
- LLM: vLLM server at http://127.0.0.1:8000/v1/chat/completions (OpenAI-compatible)
"""

import os, sys, json, signal, time, glob
from typing import List, Dict
from fastapi import FastAPI, Query
from pydantic import BaseModel
import httpx

# --- Embeddings & Reranker ---
import numpy as np
import faiss  # vector search
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Document parsing (Unstructured) ---
from unstructured.partition.auto import partition  # auto-detects file type and parses

# ---------- Config ----------
OPENAI_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
OPENAI_MODEL = os.environ.get("VLLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding model (BGE). E5 is also excellent; pick one and stay consistent.
EMBED_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_DIM = 1024  # bge-large-en-v1.5 outputs 1024-dim embeddings

# Reranker model
RERANK_NAME = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# FAISS index & metadata live in memory; we keep arrays for docs & meta
docs: List[str] = []
metas: List[Dict] = []
index = faiss.IndexFlatIP(EMBED_DIM)   # inner-product with normalized vectors

# ---------- Load models ----------
print("[boot] loading embedding model:", EMBED_NAME, "on", DEVICE, flush=True)
embed_model = SentenceTransformer(EMBED_NAME, device=DEVICE)
# Reranker = cross-encoder (scores [query, passage] pairs)
print("[boot] loading reranker:", RERANK_NAME, "on", DEVICE, flush=True)
rr_tok = AutoTokenizer.from_pretrained(RERANK_NAME)
rr_mod = AutoModelForSequenceClassification.from_pretrained(RERANK_NAME).to(DEVICE).eval()

def embed_texts(texts: List[str]) -> np.ndarray:
    """Encode texts to normalized vectors (shape: [N, EMBED_DIM])."""
    vecs = embed_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
    return vecs.astype("float32")

def add_texts(texts: List[str], meta_list: List[Dict]):
    """Add texts + metadata to FAISS and our side arrays."""
    global docs, metas, index
    embs = embed_texts(texts)
    index.add(embs)
    docs.extend(texts)
    metas.extend(meta_list)

def parse_file_to_chunks(path: str, chunk_chars: int = 1200, overlap: int = 200) -> List[Dict]:
    """
    Parse any supported file via Unstructured -> concatenate element texts -> chunk.
    Returns list of {"text": ..., "meta": {...}} ready to index.
    """
    elements = partition(filename=path)
    text = "\n".join([getattr(el, "text", "") for el in elements if getattr(el, "text", "").strip()])
    # simple fixed-size chunking by characters (tokenizers can be added later)
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_chars]
        if chunk.strip():
            chunks.append({"text": chunk, "meta": {"source": os.path.basename(path)}})
        i += (chunk_chars - overlap)
    return chunks

# ---------- FastAPI ----------
app = FastAPI(title="FinSight RAG API", version="0.1")

class QueryIn(BaseModel):
    query: str
    k: int = 12     # retrieve this many from FAISS
    topn: int = 4   # rerank to top-N

@app.get("/healthz")
def healthz():
    return {"ok": True, "docs_indexed": len(docs)}

@app.post("/ingest_dir")
def ingest_dir(path: str = Query(..., description="Directory with PDFs/DOCX/HTML")):
    """Ingest every file in a directory (non-recursive for simplicity)."""
    assert os.path.isdir(path), f"Not a directory: {path}"
    files = sorted([p for p in glob.glob(os.path.join(path, "*")) if os.path.isfile(p)])
    total = 0
    for f in files:
        try:
            items = parse_file_to_chunks(f)
            add_texts([it["text"] for it in items], [it["meta"] for it in items])
            total += len(items)
            print(f"[ingest] {os.path.basename(f)} -> {len(items)} chunks", flush=True)
        except Exception as e:
            print(f"[warn] failed to parse {f}: {e}", flush=True)
    return {"ok": True, "files": len(files), "chunks_added": total, "docs_indexed": len(docs)}

@app.post("/query")
def query(q: QueryIn):
    """Retrieve->rerank->call LLM->return citations + reasoning."""
    if len(docs) == 0:
        return {"error": "no documents indexed; call /ingest_dir first"}

    # 1) dense retrieval
    qvec = embed_texts([q.query])
    D, I = index.search(qvec, q.k)   # I: indices into docs/metas
    cands = [{"text": docs[i], "meta": metas[i]} for i in I[0]]

    # 2) cross-encoder reranking
    pairs = [(q.query, c["text"]) for c in cands]
    tok = rr_tok(pairs, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        scores = rr_mod(**tok).logits.squeeze(-1).tolist()
    ranked = [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])][:q.topn]

    # 3) build context + ask LLM via vLLM OpenAI-compatible API
    context = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(ranked)])
    system = (
        "You are an equity research assistant. Use the CONTEXT snippets; "
        "return a JSON object with keys: reasoning (step-by-step), answer (concise), "
        "citations (array of [1..N] indices pointing to CONTEXT)."
    )
    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{q.query}\n\nReturn JSON only."

    payload = {"model": OPENAI_MODEL,
               "messages":[{"role":"system","content":system},{"role":"user","content":user}],
               "temperature": 0.2}
    r = httpx.post(OPENAI_URL, json=payload, timeout=120)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]["content"]

    return {"result": msg, "docs": ranked}