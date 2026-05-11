#!/usr/bin/env python3
"""
Fetch newly submitted arXiv cs.* papers matching given keywords, then
download PDF + metadata into rag_inbox/arxiv/ for build_paper_index.py.

Differential: papers already recorded in seen.json are skipped, so this
script is safe to run every morning (intended cadence: JST 04:00 via cron).

Usage
-----
    python fetch_arxiv_cs.py \\
        --keywords "retrieval augmented generation" "tool use" \\
        --cs-category cs.CL \\
        --max-results 30 \\
        --user-agent "monst-rag-bot/0.1 daisuke.sugisawa.ts@mixi.co.jp"

Output layout
-------------
    rag_inbox/arxiv/
      ├── pdf/<arxiv_id>.pdf
      ├── metadata/<arxiv_id>.json
      └── seen.json
"""
import argparse
import json
import os
import re
import time
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import requests

ARXIV_API = "https://export.arxiv.org/api/query"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

def safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_").replace(":", "_")

def extract_arxiv_id(abs_url: str) -> str:
    return abs_url.rstrip("/").split("/")[-1]

def is_cs_paper(entry) -> bool:
    cats = [
        c.attrib.get("term", "")
        for c in entry.findall("atom:category", NS)
    ]
    return any(c.startswith("cs.") for c in cats)

def build_query(keywords, cs_category=None):
    # (all:k1 OR all:k2 OR ...) AND cat:<cs.*|specific>
    kw = " OR ".join([f'all:"{k}"' for k in keywords])
    q = f"({kw})"
    if cs_category:
        q += f" AND cat:{cs_category}"
    else:
        q += " AND cat:cs.*"
    return q

def load_seen(path: Path):
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()

def save_seen(path: Path, seen):
    path.write_text(json.dumps(sorted(seen), ensure_ascii=False, indent=2))

def download(url, out_path: Path, user_agent: str):
    headers = {"User-Agent": user_agent}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.rename(out_path)

def fetch_page(query, start, page_size, user_agent):
    params = {
        "search_query": query,
        "start": start,
        "max_results": page_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(
        ARXIV_API,
        params=params,
        headers={"User-Agent": user_agent},
        timeout=60,
    )
    r.raise_for_status()
    return ET.fromstring(r.text).findall("atom:entry", NS)


def process_entry(entry, seen, pdf_dir, meta_dir, user_agent):
    """Returns base_id if newly downloaded, None if skipped."""
    if not is_cs_paper(entry):
        return None

    abs_url = entry.findtext("atom:id", default="", namespaces=NS)
    arxiv_id = extract_arxiv_id(abs_url)
    base_id = re.sub(r"v\d+$", "", arxiv_id)

    if base_id in seen:
        return None

    title = " ".join(entry.findtext("atom:title", "", NS).split())
    summary = " ".join(entry.findtext("atom:summary", "", NS).split())
    published = entry.findtext("atom:published", "", NS)
    updated = entry.findtext("atom:updated", "", NS)

    authors = [a.findtext("atom:name", "", NS) for a in entry.findall("atom:author", NS)]
    categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", NS)]

    pdf_url = None
    for link in entry.findall("atom:link", NS):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href")
            break
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"

    sid = safe_id(base_id)
    pdf_path = pdf_dir / f"{sid}.pdf"
    meta_path = meta_dir / f"{sid}.json"

    download(pdf_url, pdf_path, user_agent)

    meta = {
        "arxiv_id": base_id,
        "versioned_id": arxiv_id,
        "title": title,
        "authors": authors,
        "summary": summary,
        "categories": categories,
        "published": published,
        "updated": updated,
        "abs_url": abs_url,
        "pdf_url": pdf_url,
        "pdf_path": str(pdf_path),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "arXiv",
        "acknowledgement": "Thank you to arXiv for use of its open access interoperability.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    seen.add(base_id)
    return base_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", nargs="+", required=True)
    ap.add_argument("--out", default="./rag_inbox/arxiv")
    ap.add_argument("--max-results", type=int, default=20,
                    help="ページサイズ (1ページあたりの API 取得件数)")
    ap.add_argument("--max-pages", type=int, default=20,
                    help="安全上限。1 run あたり最大 max_results * max_pages 件まで遡る")
    ap.add_argument("--cs-category", default=None, help="例: cs.AI, cs.CL, cs.CV")
    ap.add_argument("--user-agent", required=True, help="例: my-rag-bot/0.1 your@email.com")
    ap.add_argument("--sleep", type=float, default=3.2)
    args = ap.parse_args()

    out_dir = Path(args.out)
    pdf_dir = out_dir / "pdf"
    meta_dir = out_dir / "metadata"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    seen_path = out_dir / "seen.json"
    seen = load_seen(seen_path)

    query = build_query(args.keywords, args.cs_category)

    added = []
    pages_fetched = 0
    stop_reason = "max_pages"

    for page in range(args.max_pages):
        start = page * args.max_results
        entries = fetch_page(query, start, args.max_results, args.user_agent)
        pages_fetched += 1

        if not entries:
            stop_reason = "no_more_entries"
            break

        # Snapshot seen *before* this page so we can detect "page is fully seen"
        # — a true catch-up signal — independently of items we just added.
        seen_before_page = set(seen)

        page_new = 0
        for entry in entries:
            base_id = process_entry(entry, seen, pdf_dir, meta_dir, args.user_agent)
            if base_id is not None:
                added.append(base_id)
                page_new += 1
                time.sleep(args.sleep)

        page_ids = set()
        for entry in entries:
            abs_url = entry.findtext("atom:id", default="", namespaces=NS)
            page_ids.add(re.sub(r"v\d+$", "", extract_arxiv_id(abs_url)))

        if page_ids and page_ids.issubset(seen_before_page):
            stop_reason = "caught_up"
            break

        time.sleep(args.sleep)

    save_seen(seen_path, seen)

    print(json.dumps({
        "query": query,
        "downloaded": added,
        "count": len(added),
        "pages_fetched": pages_fetched,
        "stop_reason": stop_reason,
        "out": str(out_dir),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
