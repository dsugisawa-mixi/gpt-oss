"""
RAG search interface over D.Sugisawa's paper corpus (~/git/paper).

The corpus index lives at professor_data/index/ and is built offline by
build_paper_index.py. This module is the read-side: your_professor_server
calls search(query) per /chat request (presenting / qa stages) to retrieve
relevant excerpts that the LLM can use as grounding.

Public API
----------
    search(query, *, top_k=5, filter=None) -> list[dict]

Returned dicts have keys: {"title": str, "source": str, "text": str, "score": float}

Graceful degradation
--------------------
    If the index is missing or the embedder cannot be loaded, search()
    returns []. The server treats that as "no knowledge context" and the
    LLM falls back to slide content alone. This keeps your_professor_server
    runnable before the index is built.

Implementation status
---------------------
    Interface is fixed. The embedder + vector store wiring is marked TODO
    and currently no-ops. Wire up Qwen3-Embedding + LanceDB (or sqlite-vec)
    in _load_index() and _embed_query().
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("paper_rag")

INDEX_DIR = Path("professor_data/index")

# Runtime device for the query embedder.
# vLLM (LLM) and the TTS server already saturate the GPU, so encoding a single
# query per request is faster overall on CPU (no contention, ~100-200ms).
# Override via env if you have spare GPU.
EMBED_DEVICE = os.environ.get("PAPER_RAG_DEVICE", "cpu")

# Optional cross-encoder reranker. Re-scores the top-N dense hits.
RERANK_ENABLED = os.environ.get("PAPER_RAG_RERANK", "1") not in ("0", "false", "False", "no")
RERANKER_MODEL = os.environ.get("PAPER_RAG_RERANKER", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE = os.environ.get("PAPER_RAG_RERANKER_DEVICE", "cpu")
RERANK_TOP_K_DENSE = int(os.environ.get("PAPER_RAG_RERANK_DENSE", "10"))
# Truncate each candidate text before passing to the reranker. Long chunks
# (~600 tokens) on CPU cross-encoders cost ~500-800ms/pair; the topical
# content is almost always near the start, so first ~600 chars is enough.
RERANK_CHAR_BUDGET = int(os.environ.get("PAPER_RAG_RERANK_CHARS", "600"))

# --- Lazy-loaded embedding model + index handles ---
# These globals are protected by `_swap_lock` so reload_index() can replace
# them atomically (the server runs reload_index in a background task after
# rebuilding the index on disk; concurrent search() calls must see a
# consistent {embedder, index, meta} triple).
_embedder = None
_index = None
_index_meta: Optional[dict] = None
_load_attempted = False
_reranker = None
_reranker_attempted = False
_swap_lock = threading.Lock()


def _load_index() -> bool:
    """Lazy-load embedder and index. Returns True if both are ready.

    A failed load is cached so we don't re-spam errors per request.
    """
    global _embedder, _index, _index_meta, _load_attempted

    if _index is not None and _embedder is not None:
        return True
    if _load_attempted:
        return False
    _load_attempted = True

    if not INDEX_DIR.exists():
        logger.warning(
            "paper_rag: index dir not found: %s — run build_paper_index.py first",
            INDEX_DIR,
        )
        return False

    try:
        import json
        meta_path = INDEX_DIR / "meta.json"
        if meta_path.exists():
            _index_meta = json.loads(meta_path.read_text())
        model_name = (_index_meta or {}).get("model", "Qwen/Qwen3-Embedding-0.6B")

        from sentence_transformers import SentenceTransformer
        logger.info("paper_rag: loading embedder %s on device=%s", model_name, EMBED_DEVICE)
        _embedder = SentenceTransformer(model_name, device=EMBED_DEVICE)

        import lancedb
        db = lancedb.connect(str(INDEX_DIR / "lance"))
        if "chunks" not in db.table_names():
            logger.warning("paper_rag: 'chunks' table missing in %s", INDEX_DIR / "lance")
            return False
        _index = db.open_table("chunks")
        logger.info(
            "paper_rag: index ready (chunks=%s, dim=%s)",
            (_index_meta or {}).get("chunks", "?"),
            (_index_meta or {}).get("dim", "?"),
        )
        return True
    except Exception:
        logger.exception("paper_rag: failed to load index")
        return False


def _load_reranker() -> bool:
    """Lazy-load the cross-encoder reranker. Cached failure flag avoids spam."""
    global _reranker, _reranker_attempted
    if _reranker is not None:
        return True
    if _reranker_attempted or not RERANK_ENABLED:
        return False
    _reranker_attempted = True
    try:
        from sentence_transformers import CrossEncoder
        logger.info(
            "paper_rag: loading reranker %s on device=%s",
            RERANKER_MODEL, RERANKER_DEVICE,
        )
        _reranker = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)
        return True
    except Exception:
        logger.exception("paper_rag: failed to load reranker; falling back to dense-only")
        return False


def preload() -> bool:
    """Eagerly load the embedder + index (and reranker, if enabled). Call at
    server startup so the first /chat request doesn't pay the ~9s cold-load tax.
    Returns True iff the index is ready (reranker is best-effort)."""
    ok = _load_index()
    if RERANK_ENABLED:
        _load_reranker()
    return ok


def _snapshot():
    """Return the current (embedder, index, meta) triple under the swap
    lock so a concurrent reload_index() can't tear the state mid-search."""
    with _swap_lock:
        return _embedder, _index, _index_meta


def _embed_query(embedder, query: str):
    """Embed a query string. Returns a numpy vector or None if unavailable."""
    if embedder is None:
        return None
    vec = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )
    return vec[0]


def reload_index() -> bool:
    """Rebuild in-memory state from the current on-disk index, then swap
    atomically. Caller (server) typically runs this in a thread executor
    after build_paper_index.py finishes regenerating the LanceDB table.

    Returns True on successful swap. On failure the previous state is
    left intact and the caller can retry. The old embedder / index handles
    are released as soon as their last reference (held by in-flight
    search() calls) drops.
    """
    global _embedder, _index, _index_meta, _load_attempted

    if not INDEX_DIR.exists():
        logger.warning("paper_rag: reload_index aborted — %s missing", INDEX_DIR)
        return False

    # Build new state in locals — the live state stays unchanged until the
    # swap below succeeds.
    try:
        import json as _json
        meta_path = INDEX_DIR / "meta.json"
        new_meta = _json.loads(meta_path.read_text()) if meta_path.exists() else {}
        model_name = new_meta.get("model", "Qwen/Qwen3-Embedding-0.6B")

        from sentence_transformers import SentenceTransformer
        logger.info("paper_rag: reload — loading embedder %s on %s",
                    model_name, EMBED_DEVICE)
        new_embedder = SentenceTransformer(model_name, device=EMBED_DEVICE)

        import lancedb
        db = lancedb.connect(str(INDEX_DIR / "lance"))
        if "chunks" not in db.table_names():
            logger.warning("paper_rag: reload — 'chunks' table missing")
            return False
        new_index = db.open_table("chunks")
    except Exception:
        logger.exception("paper_rag: reload_index build failed; keeping old state")
        return False

    # Atomic swap. Old refs drop here; in-flight search() calls keep their
    # snapshot via _snapshot() until they return, then the old embedder /
    # index get garbage-collected.
    with _swap_lock:
        _embedder = new_embedder
        _index = new_index
        _index_meta = new_meta
        _load_attempted = True

    logger.info("paper_rag: index reloaded (chunks=%s, dim=%s)",
                new_meta.get("chunks", "?"), new_meta.get("dim", "?"))
    return True


def get_index_meta() -> Optional[dict]:
    """Return a shallow copy of the current index meta, or None if unloaded.
    Acquires the swap lock so a concurrent reload_index() can't tear state.
    Caller-mutable copy."""
    with _swap_lock:
        return dict(_index_meta) if _index_meta else None


def embed_texts(texts: list[str]):
    """Encode a batch of strings with the same embedder used for queries,
    returning an L2-normalized numpy array of shape (n, dim) or None if
    the embedder is unavailable.

    FACR's novelty term needs to compare new evidence against the prior
    evidence pool by cosine similarity, and reusing the live SentenceTransformer
    avoids loading a second model just for this. Empty list returns None too
    (callers fall back to a token-overlap heuristic in that case).
    """
    if not texts:
        return None
    if not _load_index():
        return None
    embedder, _index, _meta = _snapshot()
    if embedder is None:
        return None
    try:
        return embedder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
    except Exception:
        logger.exception("paper_rag.embed_texts failed")
        return None


def _filter_to_sql(filter: dict) -> str:
    """Convert a flat {key: value} dict into a LanceDB WHERE clause.

    Strings are quoted, bools/numbers passed through. AND-joined.
    """
    parts: list[str] = []
    for k, v in filter.items():
        if isinstance(v, str):
            escaped = v.replace("'", "''")
            parts.append(f"{k} = '{escaped}'")
        elif isinstance(v, bool):
            parts.append(f"{k} = {str(v).lower()}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k} = {v}")
        elif isinstance(v, (list, tuple)) and v:
            quoted = ", ".join(
                f"'{x.replace(chr(39), chr(39) * 2)}'" if isinstance(x, str) else str(x)
                for x in v
            )
            parts.append(f"{k} IN ({quoted})")
        else:
            logger.warning("paper_rag: skipping unsupported filter %s=%r", k, v)
    return " AND ".join(parts)


def search(
    query: str,
    *,
    top_k: int = 5,
    filter: Optional[dict] = None,
) -> list[dict]:
    """Vector search over the paper corpus.

    Args:
        query:   free-text search string. For presenting it is typically
                 slide.title + bullets joined; for qa it includes the
                 audience question.
        top_k:   how many top hits to return.
        filter:  metadata constraints applied at the DB level, e.g.
                 {"doc_type": "paper"} or {"topic": ["paper", "preprints.org-quic"]}.
                 None = no filter.

    Returns:
        list of {title, source, text, score} dicts. Empty list if the
        index is unavailable or the query is empty.
    """
    if not query or not query.strip():
        return []
    if not _load_index():
        return []

    # Snapshot the live triple under the swap lock so an in-flight
    # reload_index() can't replace one of {embedder, index, meta} between
    # the embed call and the search call.
    embedder, index, _meta = _snapshot()
    if embedder is None or index is None:
        return []

    qvec = _embed_query(embedder, query)
    if qvec is None:
        return []

    # Pull a wider candidate pool when reranking; otherwise just top_k.
    dense_limit = max(top_k, RERANK_TOP_K_DENSE) if RERANK_ENABLED else max(top_k, 1)

    try:
        q = index.search(qvec).limit(dense_limit)
        if filter:
            where = _filter_to_sql(filter)
            if where:
                q = q.where(where)
        rows = q.to_list()
    except Exception:
        logger.exception("paper_rag: search failed")
        return []

    if not rows:
        return []

    # Optional cross-encoder rerank.
    use_rerank = RERANK_ENABLED and len(rows) > top_k and _load_reranker()
    if use_rerank:
        try:
            pairs = [
                (query, (r.get("text") or "")[:RERANK_CHAR_BUDGET])
                for r in rows
            ]
            ce_scores = _reranker.predict(pairs)
            ranked = sorted(
                zip(rows, ce_scores),
                key=lambda pair: float(pair[1]),
                reverse=True,
            )[:top_k]
            return [
                {
                    "title": r.get("title", "untitled"),
                    "source": r.get("source_path", ""),
                    "text": r.get("text", ""),
                    "score": float(s),
                }
                for r, s in ranked
            ]
        except Exception:
            logger.exception("paper_rag: rerank failed; falling back to dense order")

    out: list[dict] = []
    for r in rows[:top_k]:
        # LanceDB L2 distance → higher-is-better score.
        dist = r.get("_distance")
        score = (1.0 - float(dist)) if dist is not None else 0.0
        out.append({
            "title": r.get("title", "untitled"),
            "source": r.get("source_path", ""),
            "text": r.get("text", ""),
            "score": score,
        })
    return out
