#!/usr/bin/env bash
# Daily RAG index rebuild — 06:00 JST.
# 1) Mirror newly-fetched arXiv PDFs into the curated paper corpus
#    (cp -np: never overwrite; arXiv IDs are immutable per version).
# 2) Rebuild the LanceDB index consumed by your_professor_server / paper_rag.
#    The `professor` conda env (pymupdf / sentence_transformers / lancedb) lives
#    in the gpt-oss-llm image; host base env lacks all three. The running
#    gpt-oss-llm container holds the LLM weights on the GPU though, so embedding
#    OOMs if we just `docker exec` into it. Sequence: graceful stop → one-off
#    `compose run` against the same image+mounts → force-recreate the service.
set -euo pipefail

ARXIV_PDF_DIR="$HOME/git/gpt-oss/rag_inbox/arxiv/pdf"
ARXIV_META_DIR="$HOME/git/gpt-oss/rag_inbox/arxiv/metadata"
DEST_DIR="$HOME/git/paper/external-pdf-for-rag"
COMPOSE_DIR="$HOME/git/Qwen3-TTS-streamingmx"
SERVICE="gpt-oss-llm"

# Tempfiles holding the pre/post PDF lists and the resulting added-IDs list.
# Living under a single dir lets us mount it into the indexer container with
# one -v flag.
DIFF_DIR=$(mktemp -d -t rag_rebuild_XXXX)
trap 'rm -rf "$DIFF_DIR"' EXIT INT TERM
BEFORE_IDS="$DIFF_DIR/before_ids.txt"
AFTER_IDS="$DIFF_DIR/after_ids.txt"
ADDED_IDS="$DIFF_DIR/added_ids.txt"

# arXiv IDs are the PDF basenames sans `.pdf` and sans any `vN` suffix.
list_arxiv_ids() {
    local dir="$1"
    if compgen -G "$dir/*.pdf" > /dev/null; then
        find "$dir" -maxdepth 1 -name '*.pdf' -printf '%f\n' \
            | sed -E 's/\.pdf$//; s/v[0-9]+$//' \
            | sort -u
    fi
}

echo "===== $(date -Iseconds) start rag index rebuild ====="

mkdir -p "$DEST_DIR"
list_arxiv_ids "$DEST_DIR" > "$BEFORE_IDS"

if compgen -G "$ARXIV_PDF_DIR/*.pdf" > /dev/null; then
    before=$(find "$DEST_DIR" -maxdepth 1 -name '*.pdf' | wc -l)
    cp -np "$ARXIV_PDF_DIR"/*.pdf "$DEST_DIR/"
    after=$(find "$DEST_DIR" -maxdepth 1 -name '*.pdf' | wc -l)
    echo "copied: dest pdf count $before -> $after"
else
    echo "no PDFs in $ARXIV_PDF_DIR; skipping copy"
fi

list_arxiv_ids "$DEST_DIR" > "$AFTER_IDS"
# Set difference: IDs present after but not before.
comm -13 "$BEFORE_IDS" "$AFTER_IDS" > "$ADDED_IDS"
added_count=$(wc -l < "$ADDED_IDS" | tr -d ' ')
echo "added arxiv IDs since last rebuild: $added_count"

cd "$COMPOSE_DIR"

# Always bring the LLM service back, even if the indexer fails halfway —
# otherwise the API stays down until someone notices.
restore_service() {
    echo "----- $(date -Iseconds) restoring $SERVICE -----"
    docker compose up -d --force-recreate "$SERVICE"
}
trap restore_service EXIT

echo "----- $(date -Iseconds) stopping $SERVICE to free GPU -----"
docker compose stop "$SERVICE"

# `compose run` reuses the service's image, mounts (/app/paper, /app/professor_data),
# and GPU reservation. --rm cleans up; --no-deps avoids touching qwen3-tts / tunnel;
# -T disables TTY allocation for cron.
#
# Two extra mounts feed the daily-additions step that runs after the build:
#   /host/arxiv_meta : per-paper arXiv metadata json (title + abstract)
#   /host/rebuild    : the tempdir holding added_ids.txt (and friends)
# record_daily_additions.py is baked into the image at /app/.
# We run build_paper_index.py and record_daily_additions.py in one container so
# we only pay the start-up cost once.
docker compose run --rm --no-deps -T \
    -v "$ARXIV_META_DIR:/host/arxiv_meta:ro" \
    -v "$DIFF_DIR:/host/rebuild:ro" \
    "$SERVICE" \
    bash -c '
        set -e
        conda run --no-capture-output -n professor \
            python /app/build_paper_index.py \
                --paper-dir /app/paper \
                --output /app/professor_data/index
        # daily additions: pull built_at out of the meta.json that
        # build_paper_index.py just wrote so the GUI entry links back to
        # this specific build, not wall-clock-now.
        BUILT_AT=$(python -c "import json,sys; print(json.load(open(\"/app/professor_data/index/meta.json\")).get(\"built_at\",\"\"))")
        python /app/record_daily_additions.py \
            --metadata-dir /host/arxiv_meta \
            --added-ids-file /host/rebuild/added_ids.txt \
            --output /app/professor_data/index/daily_additions.json \
            --built-at "${BUILT_AT:-0}"
    '

echo "===== $(date -Iseconds) done rag index rebuild ====="
