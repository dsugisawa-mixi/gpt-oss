#!/usr/bin/env bash
# Daily container refresh — 07:00 JST.
# Stop then recreate so the server reloads the freshly-built RAG index.
set -euo pipefail

QWEN_DIR="$HOME/git/Qwen3-TTS-streamingmx"

echo "===== $(date -Iseconds) start qwen3-tts restart ====="

cd "$QWEN_DIR"
/usr/bin/docker compose stop
/usr/bin/docker compose up -d --force-recreate

echo "===== $(date -Iseconds) done qwen3-tts restart ====="
/usr/bin/docker compose ps
