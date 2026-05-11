#!/usr/bin/env python3
"""Record one day's worth of newly-indexed papers as a bullet list.

Called from cron_rebuild_rag_index.sh right after build_paper_index.py
succeeds. Reads the arXiv metadata json files that fetch_arxiv_cs.py
already wrote (title, summary, authors, ...) for each newly-added ID
and appends a YYYYMMDD-keyed entry to daily_additions.json next to the
LanceDB index. That file is then consumed by your_professor_server's
/api/lab/summary so the LAB list GUI can show "what knowledge entered
this corpus, when".

No LLM, no network; pure stdlib so it runs in any python the cron has.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("record_daily_additions")


# Strip math/LaTeX-ish whitespace noise that arXiv abstracts carry over from PDFs.
_WS = re.compile(r"\s+")
# A passable sentence boundary for English/Japanese abstracts. arXiv abstracts
# are overwhelmingly English so the "[.!?] + space + capital" heuristic is fine;
# we cap by char length anyway so a bad split just produces a slightly long brief.
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def first_sentence(text: str, max_chars: int = 220) -> str:
    """Return the first sentence of an abstract, bounded by max_chars.

    Falls back to a hard char truncate when no sentence boundary is found
    within the cap (typical for abstracts that open with a multi-clause
    sentence)."""
    t = _WS.sub(" ", (text or "").strip())
    if not t:
        return ""
    parts = _SENT_END.split(t, maxsplit=1)
    head = parts[0]
    if len(head) > max_chars:
        head = head[: max_chars - 1].rstrip() + "…"
    return head


def load_metadata(metadata_dir: Path, arxiv_id: str) -> dict | None:
    """Load metadata/<id>.json written by fetch_arxiv_cs.py.

    Returns None if the file is missing — caller decides whether to skip
    the entry or fall back to a thinner record."""
    p = metadata_dir / f"{arxiv_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        logger.exception("failed to parse %s", p)
        return None


def build_item(arxiv_id: str, metadata_dir: Path) -> dict:
    """One bullet entry. Always includes arxiv_id so the GUI can link out
    even when metadata is missing for an older paper."""
    meta = load_metadata(metadata_dir, arxiv_id) or {}
    title = (meta.get("title") or "").strip() or f"(no title for {arxiv_id})"
    brief = first_sentence(meta.get("summary") or "")
    item: dict = {"arxiv_id": arxiv_id, "title": title}
    if brief:
        item["brief"] = brief
    # Keep authors short — first author + et al. on long lists. The GUI
    # doesn't surface this yet but it's cheap to include for future use.
    authors = meta.get("authors") or []
    if authors:
        item["authors_short"] = (
            authors[0] if len(authors) == 1 else f"{authors[0]} et al."
        )
    return item


def read_id_list(path: Path) -> list[str]:
    """One arxiv ID per line; blank lines and #-comments ignored.
    PDF filenames (e.g. "2408.01791.pdf" or "2408.01791v3.pdf") are
    accepted too — the .pdf and trailing version suffix are stripped so
    the result matches metadata filenames."""
    ids: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().endswith(".pdf"):
            s = s[:-4]
        # Drop a trailing version tag like "v1", "v12" to canonicalize
        # — metadata/ is keyed by bare arxiv_id.
        s = re.sub(r"v\d+$", "", s)
        if s and s not in seen:
            seen.add(s)
            ids.append(s)
    return ids


def merge_into(store: dict, date_key: str, items: list[dict], built_at: int | None) -> None:
    """Replace (not append) the entry for date_key.

    Re-running on the same day intentionally overwrites: a same-day
    second rebuild reflects the cumulative state, not a stale partial
    one, and avoids unbounded duplication if cron retries."""
    store[date_key] = {
        "built_at": built_at,
        "added_count": len(items),
        "items": items,
    }


def prune(store: dict, keep_days: int) -> None:
    """Trim entries older than keep_days (lexicographic compare works
    because YYYYMMDD is monotonic). Set keep_days<=0 to disable."""
    if keep_days <= 0:
        return
    keys = sorted(store.keys())
    if len(keys) <= keep_days:
        return
    cutoff = keys[-keep_days]
    for k in keys:
        if k < cutoff:
            store.pop(k, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-dir", type=Path, required=True,
                    help="dir containing <arxiv_id>.json files (rag_inbox/arxiv/metadata)")
    ap.add_argument("--added-ids-file", type=Path, required=True,
                    help="text file listing arxiv IDs added in this rebuild, one per line")
    ap.add_argument("--output", type=Path, required=True,
                    help="path to daily_additions.json (will be created or merged)")
    ap.add_argument("--date", default=None,
                    help="YYYYMMDD override; defaults to today in local TZ")
    ap.add_argument("--built-at", type=int, default=None,
                    help="unix timestamp of the index build (links GUI entry to the build)")
    ap.add_argument("--keep-days", type=int, default=90,
                    help="prune entries older than this many days (0 disables)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.metadata_dir.is_dir():
        logger.error("metadata-dir not found: %s", args.metadata_dir)
        return 2
    if not args.added_ids_file.exists():
        logger.error("added-ids-file not found: %s", args.added_ids_file)
        return 2

    ids = read_id_list(args.added_ids_file)
    if not ids:
        logger.info("no added IDs — skipping daily_additions write")
        return 0

    date_key = args.date or datetime.now().strftime("%Y%m%d")
    built_at = args.built_at if args.built_at is not None else int(
        datetime.now(timezone.utc).timestamp()
    )

    items = [build_item(i, args.metadata_dir) for i in ids]

    store: dict = {}
    if args.output.exists():
        try:
            store = json.loads(args.output.read_text())
            if not isinstance(store, dict):
                logger.warning("existing %s is not a dict; replacing", args.output)
                store = {}
        except Exception:
            logger.exception("failed to read existing %s; replacing", args.output)
            store = {}

    merge_into(store, date_key, items, built_at)
    prune(store, args.keep_days)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2))
    tmp.replace(args.output)

    # Echo to stdout so cron_rag_index.log carries the human-readable summary
    # — that's the "通知" path until we wire a real notifier.
    print(f"===== daily additions ({date_key}): {len(items)} paper(s) =====")
    for it in items:
        line = f"- {it['title']} ({it['arxiv_id']})"
        if it.get("brief"):
            line += f": {it['brief']}"
        print(line)

    logger.info("wrote %d items for %s to %s", len(items), date_key, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
