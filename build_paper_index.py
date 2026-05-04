"""
Build a vector index over D.Sugisawa's paper corpus for paper_rag.search().

Walk the paper directory, extract text from PDF / Markdown / TeX / txt,
chunk it, embed each chunk, and persist to a vector store + sidecar
manifest. Run this offline whenever the corpus changes.

Usage
-----
    # Cheap discovery + chunk pass to validate the walk and chunker:
    python build_paper_index.py --dry-run

    # Full build (once Qwen3-Embedding + LanceDB are wired up):
    python build_paper_index.py \\
        --paper-dir ~/git/paper \\
        --output professor_data/index \\
        --model Qwen/Qwen3-Embedding-0.6B

Output layout
-------------
    professor_data/index/
      ├── lance/             # LanceDB table "chunks" (vec + metadata)
      └── meta.json          # build manifest: model, dim, doc count, built_at

Implementation status
---------------------
    File walk + Markdown/TeX/txt reading + paragraph-aware chunking are
    implemented (cheap, no extra deps). PDF extraction, embedding, and
    vector-DB persistence are TODO — see the inline markers in
    extract_text() and main().
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("build_paper_index")

DEFAULT_PAPER_DIR = Path.home() / "git" / "paper"
# Per-user paper uploads land under professor_data/uploads/<job>/*.pdf,
# alongside the generated slides + script. Including this dir in the
# walk lets the freshly-uploaded paper feed RAG retrieval as soon as
# the index is rebuilt (triggered by your_professor_server after the
# upload+generation pipeline finishes).
DEFAULT_UPLOAD_DIR = Path("professor_data/uploads")
DEFAULT_OUTPUT = Path("professor_data/index")
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Directory names anywhere in the path that disqualify a file.
EXCLUDE_DIRNAMES = {
    "figures", "image", "drawio",
    "node_modules", ".git", "__pycache__", ".venv", "venv",
}

INCLUDE_EXTS = {".pdf", ".md", ".tex", ".txt"}

# Cheap char-based proxy for token budget. Roughly 4 chars / token for
# mixed Japanese/English; refine once we plug in the real tokenizer.
CHUNK_TOKENS = 600
CHUNK_OVERLAP = 100
CHARS_PER_TOKEN = 4


# =====================================================================
# Discovery
# =====================================================================

def discover_files(paper_dir: Path) -> list[Path]:
    """Walk paper_dir and return candidate files filtered by ext + dir name."""
    out: list[Path] = []
    for p in paper_dir.rglob("*"):
        if p.is_dir():
            continue
        if any(part in EXCLUDE_DIRNAMES for part in p.parts):
            continue
        if p.suffix.lower() in INCLUDE_EXTS:
            out.append(p)
    return out


# =====================================================================
# Text extraction
# =====================================================================

def extract_text(path: Path) -> str:
    """Extract plain text. PDF via PyMuPDF; MD/TeX/txt verbatim."""
    suf = path.suffix.lower()
    if suf == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed; skipping %s", path)
            return ""
        try:
            with fitz.open(path) as doc:
                return "\n\n".join(page.get_text() for page in doc)
        except Exception as e:
            logger.warning("PDF read failed %s: %s", path, e)
            return ""
    if suf in (".md", ".tex", ".txt"):
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("read failed %s: %s", path, e)
            return ""
    return ""


# =====================================================================
# Chunking
# =====================================================================

def chunk_text(
    text: str,
    max_tokens: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Paragraph-aware chunking by approximate token count.

    Splits on blank lines, packs paragraphs into ~max_tokens-sized chunks,
    and keeps a small tail-overlap between consecutive chunks so context
    survives the boundary.
    """
    if not text or not text.strip():
        return []

    char_budget = max_tokens * CHARS_PER_TOKEN
    char_overlap = overlap * CHARS_PER_TOKEN

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
            continue
        if len(buf) + len(p) + 2 <= char_budget:
            buf += "\n\n" + p
        else:
            chunks.append(buf)
            tail = buf[-char_overlap:] if char_overlap else ""
            buf = (tail + "\n\n" + p).strip() if tail else p
    if buf:
        chunks.append(buf)
    return chunks


# =====================================================================
# Metadata
# =====================================================================

def classify_doc(path: Path, paper_dir: Path) -> dict:
    """Derive light metadata from the path (no file read)."""
    rel = path.relative_to(paper_dir)
    parts = rel.parts
    top = parts[0] if parts else ""

    if top == "paper":
        doc_type = "paper"
    elif top.startswith("preprints."):
        doc_type = "preprint"
    elif top == "patent":
        doc_type = "patent"
    elif top == "myboy":
        doc_type = "memo"
    elif top == "tools":
        doc_type = "tool"
    else:
        doc_type = "other"

    return {
        "doc_id": str(rel),
        "source_path": str(path),
        "doc_type": doc_type,
        "topic": top,
        "title": path.stem,
        "ext": path.suffix.lower().lstrip("."),
    }


# =====================================================================
# Main
# =====================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Build the paper RAG index")
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR,
                        help="extra dir to walk (uploaded papers); ignored if missing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None,
                        help="torch device: 'cuda', 'cpu', etc. Auto-detect if omitted.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="discover + chunk only; skip embedding & persist")
    args = parser.parse_args()

    if not args.paper_dir.exists():
        raise SystemExit(f"paper-dir not found: {args.paper_dir}")

    started = time.time()
    # Walk the primary corpus + the optional upload dir. Each file is paired
    # with the base it was discovered under so classify_doc derives the
    # right relative path. Upload files end up with doc_type="other" and
    # topic=<job_id> since they don't match any of the known top-level
    # buckets in paper-dir — that's fine, they're still searchable.
    discovered: list[tuple[Path, Path]] = [
        (f, args.paper_dir) for f in discover_files(args.paper_dir)
    ]
    logger.info("discovered %d candidate files in %s", len(discovered), args.paper_dir)
    if args.upload_dir and args.upload_dir.exists():
        upload_files = discover_files(args.upload_dir)
        discovered += [(f, args.upload_dir) for f in upload_files]
        logger.info("+ %d files from %s", len(upload_files), args.upload_dir)

    by_type: dict[str, int] = {}
    by_ext: dict[str, int] = {}
    total_chunks = 0
    skipped_empty = 0

    records: list[dict] = []
    for f, base in discovered:
        meta = classify_doc(f, base)
        by_type[meta["doc_type"]] = by_type.get(meta["doc_type"], 0) + 1
        by_ext[meta["ext"]] = by_ext.get(meta["ext"], 0) + 1

        text = extract_text(f)
        chunks = chunk_text(text)
        if not chunks:
            skipped_empty += 1
            continue

        for i, chunk in enumerate(chunks):
            rec = dict(meta)
            rec["chunk_idx"] = i
            rec["text"] = chunk
            records.append(rec)
        total_chunks += len(chunks)

    logger.info(
        "summary: files=%d chunks=%d skipped_empty=%d by_type=%s by_ext=%s",
        len(discovered), total_chunks, skipped_empty, by_type, by_ext,
    )

    if args.dry_run:
        logger.info("dry-run: skipping embedding + persist")
        return
    if not records:
        logger.warning("no records to index; aborting")
        return

    # --- Embed ---
    from sentence_transformers import SentenceTransformer
    import lancedb

    logger.info("loading embedding model: %s (device=%s)", args.model, args.device or "auto")
    model = SentenceTransformer(args.model, device=args.device)
    texts = [r["text"] for r in records]
    logger.info("encoding %d chunks (batch=%d) ...", len(texts), args.batch_size)
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    dim = int(vecs.shape[1])
    for r, v in zip(records, vecs):
        r["vector"] = v.tolist()

    # --- Persist ---
    args.output.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(args.output / "lance"))
    if "chunks" in db.table_names():
        db.drop_table("chunks")
    db.create_table("chunks", data=records)

    manifest = {
        "model": args.model,
        "dim": dim,
        "files": len(discovered),
        "chunks": len(records),
        "by_type": by_type,
        "by_ext": by_ext,
        "built_at": int(time.time()),
        "build_seconds": round(time.time() - started, 1),
    }
    (args.output / "meta.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    logger.info("persisted index to %s (dim=%d, chunks=%d)", args.output, dim, len(records))


if __name__ == "__main__":
    main()
