#!/usr/bin/env python3
"""
Fetch newly submitted arXiv cs.* papers via the per-category Atom feeds,
then download PDF + metadata into rag_inbox/arxiv/ for build_paper_index.py.

Why RSS, not the API?
---------------------
The arXiv /api/query endpoint rate-limits our shared NAT IP aggressively
(429 even on simple `cat:cs.NI` queries after a few hits). The Atom feeds
at https://rss.arxiv.org/atom/<category> are server-cached, refresh once
per day, and are the documented mechanism for daily polling.

Filtering is therefore entirely client-side:
- keyword: substring match against title + summary (case-insensitive)
- date:    `published` compared against --date-from / --date-to / --recent-days

Differential: papers already recorded in seen.json are skipped, so this
script is safe to run every morning (intended cadence: JST 04:00 via cron).

Usage
-----
    python fetch_arxiv_cs.py \\
        --cs-categories cs.NI cs.DC cs.CR cs.SE \\
        --keywords "QUIC" "WebRTC" "HTTP/3" \\
        --recent-days 3 \\
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
import re
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET

import requests

ARXIV_RSS_BASE = "https://rss.arxiv.org/atom/"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "dc": "http://purl.org/dc/elements/1.1/",
}

def safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_").replace(":", "_")

def extract_arxiv_id(abs_url: str) -> str:
    # arxiv RSS sometimes uses "oai:arXiv.org:2605.12345v1"
    if abs_url.startswith("oai:arXiv.org:"):
        return abs_url.split(":", 2)[-1]
    return abs_url.rstrip("/").split("/")[-1]

def is_cs_paper(entry) -> bool:
    cats = [
        c.attrib.get("term", "")
        for c in entry.findall("atom:category", NS)
    ]
    return any(c.startswith("cs.") for c in cats)

def parse_date_arg(s):
    """YYYY-MM-DD or YYYYMMDD → date object."""
    s = s.replace("-", "")
    return datetime.strptime(s[:8], "%Y%m%d").replace(tzinfo=timezone.utc)


def _keyword_pattern(keyword):
    # Word-boundary match so "QUIC" does not hit "quickly"/"QuickFPS"/"arquicanedo".
    # Lookarounds (not \b) so punctuation-bearing tokens like "HTTP/3" still match.
    return re.compile(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", re.IGNORECASE)


def matches_keywords(title, summary, keywords):
    if not keywords:
        return True
    hay = title + " " + summary
    return any(_keyword_pattern(k).search(hay) for k in keywords)

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


def fetch_feed(category, user_agent, max_retries=3):
    url = ARXIV_RSS_BASE + category
    backoff = 10
    for attempt in range(max_retries + 1):
        r = requests.get(
            url,
            headers={"User-Agent": user_agent},
            timeout=60,
        )
        if r.status_code == 429:
            if attempt == max_retries:
                r.raise_for_status()
            wait = int(r.headers.get("Retry-After", backoff))
            print(f"  429 on {category}, waiting {wait}s (attempt {attempt+1}/{max_retries})",
                  flush=True)
            time.sleep(wait)
            backoff = min(backoff * 2, 120)
            continue
        r.raise_for_status()
        return ET.fromstring(r.text).findall("atom:entry", NS)
    return []


_TOO_OLD = "__too_old__"
_TOO_NEW = "__too_new__"
_NO_KEYWORD = "__no_keyword__"


def process_entry(entry, seen, pdf_dir, meta_dir, user_agent,
                  date_from=None, date_to=None, keywords=None):
    """Returns base_id if newly downloaded, _TOO_OLD/_TOO_NEW for date
    out-of-range, _NO_KEYWORD if keyword filter rejects, None if skipped
    for other reasons."""
    if not is_cs_paper(entry):
        return None

    abs_url = entry.findtext("atom:id", default="", namespaces=NS)
    arxiv_id = extract_arxiv_id(abs_url)
    base_id = re.sub(r"v\d+$", "", arxiv_id)

    published = entry.findtext("atom:published", "", NS)
    updated = entry.findtext("atom:updated", "", NS)

    # RSS feeds may omit <published>; fall back to <updated>.
    date_str = published or updated
    if date_str and (date_from or date_to):
        pub_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if date_to and pub_dt > date_to:
            return _TOO_NEW
        if date_from and pub_dt < date_from:
            return _TOO_OLD

    if base_id in seen:
        return None

    title = " ".join(entry.findtext("atom:title", "", NS).split())
    summary = " ".join(entry.findtext("atom:summary", "", NS).split())

    if not matches_keywords(title, summary, keywords):
        return _NO_KEYWORD

    # RSS feed uses <dc:creator> (single comma-separated string), not <atom:author>.
    authors = []
    creator = entry.findtext("dc:creator", "", NS)
    if creator:
        authors = [a.strip() for a in creator.split(",") if a.strip()]
    else:
        authors = [a.findtext("atom:name", "", NS) for a in entry.findall("atom:author", NS)]

    categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", NS)]

    # Prefer <link rel="alternate" type="text/html"> for the human-readable abs URL.
    html_url = ""
    pdf_url = None
    for link in entry.findall("atom:link", NS):
        rel = link.attrib.get("rel", "")
        typ = link.attrib.get("type", "")
        link_title = link.attrib.get("title", "")
        href = link.attrib.get("href", "")
        if link_title == "pdf" or typ == "application/pdf":
            pdf_url = href
        elif rel == "alternate" and typ == "text/html":
            html_url = href
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
    if not html_url:
        html_url = f"https://arxiv.org/abs/{base_id}"

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
        "abs_url": html_url,
        "feed_id": abs_url,
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
    ap.add_argument("--cs-categories", nargs="+", required=True,
                    help="例: cs.NI cs.DC cs.CR cs.SE")
    ap.add_argument("--keywords", nargs="+", default=None,
                    help="title/summary に対するクライアント側フィルタ。未指定なら全件通過")
    ap.add_argument("--out", default="./rag_inbox/arxiv")
    ap.add_argument("--date-from", default=None,
                    help="published 下限 (YYYY-MM-DD or YYYYMMDD)")
    ap.add_argument("--date-to", default=None,
                    help="published 上限 (YYYY-MM-DD or YYYYMMDD)")
    ap.add_argument("--recent-days", type=int, default=None,
                    help="--date-from 未指定時、直近N日を自動設定 (例: 3)")
    ap.add_argument("--user-agent", required=True, help="例: my-rag-bot/0.1 your@email.com")
    ap.add_argument("--sleep", type=float, default=3.2,
                    help="フィード取得・PDF DL の間隔(秒)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    pdf_dir = out_dir / "pdf"
    meta_dir = out_dir / "metadata"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    seen_path = out_dir / "seen.json"
    seen = load_seen(seen_path)

    if args.date_from:
        date_from = parse_date_arg(args.date_from)
    elif args.recent_days:
        date_from = datetime.now(timezone.utc) - timedelta(days=args.recent_days)
    else:
        date_from = None

    date_to = parse_date_arg(args.date_to) if args.date_to else None

    added = []
    feed_results = []

    for category in args.cs_categories:
        entries = fetch_feed(category, args.user_agent)
        n_new = n_skip_seen = n_no_kw = n_old = n_too_new = 0

        for entry in entries:
            result = process_entry(entry, seen, pdf_dir, meta_dir,
                                   args.user_agent, date_from, date_to,
                                   args.keywords)
            if result == _TOO_OLD:
                n_old += 1
            elif result == _TOO_NEW:
                n_too_new += 1
            elif result == _NO_KEYWORD:
                n_no_kw += 1
            elif result is None:
                n_skip_seen += 1
            else:
                added.append(result)
                n_new += 1
                time.sleep(args.sleep)

        feed_results.append({
            "category": category,
            "total_entries": len(entries),
            "downloaded": n_new,
            "skipped_seen": n_skip_seen,
            "filtered_no_keyword": n_no_kw,
            "filtered_too_old": n_old,
            "filtered_too_new": n_too_new,
        })
        time.sleep(args.sleep)

    save_seen(seen_path, seen)

    print(json.dumps({
        "feeds": feed_results,
        "downloaded": added,
        "count": len(added),
        "out": str(out_dir),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
