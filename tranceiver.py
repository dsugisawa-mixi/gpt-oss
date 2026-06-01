"""
Opus linephone relay server.

POST /api/device/linephone/  -- device -> server: send Opus frames
GET  /api/device/linephone/  -- server -> device: receive Opus stream

Wire format (both directions):
    [BE16 length][opus packet bytes]  repeated per frame

GET params:
    start      -- position id to resume from (0 = from current head)
    device_id  -- caller's device id; frames from this sender are skipped
    echo       -- 1 to include own frames (debug echo mode)

Usage:
    pip install fastapi uvicorn
    python tranceiver.py [--port 8891]
"""

import argparse
import asyncio
import base64
import logging
import os
import struct
import threading

# opuslib needs libopus on the library path.  On macOS with Homebrew:
_opus_lib_hint = "/opt/homebrew/lib"
if os.path.isdir(_opus_lib_hint):
    _cur = os.environ.get("DYLD_LIBRARY_PATH", "")
    if _opus_lib_hint not in _cur:
        os.environ["DYLD_LIBRARY_PATH"] = f"{_opus_lib_hint}:{_cur}" if _cur else _opus_lib_hint
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPUS_SR = int(os.environ.get("OPUS_SR", "16000"))
OPUS_CHANNELS = int(os.environ.get("OPUS_CHANNELS", "1"))
OPUS_FRAME_MS = int(os.environ.get("OPUS_FRAME_MS", "20"))
OPUS_FRAME_SAMPLES = OPUS_SR * OPUS_FRAME_MS // 1000
OPUS_BITRATE = int(os.environ.get("OPUS_BITRATE", "16000"))

# Ring buffer capacity (number of frames kept in memory)
RING_CAPACITY = int(os.environ.get("RING_CAPACITY", "50000"))  # ~16 min at 20ms/frame

# Debug: save incoming Opus packets to files under this directory (empty = off)
DEBUG_DUMP_DIR = os.environ.get("DEBUG_DUMP_DIR", "").strip()
_dump_counter: int = 0

INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "")

LAB_ID = os.environ.get("LAB_ID", "lab-linephone").strip()
LAB_NAME = os.environ.get("LAB_NAME", "Linephone (Opus)")
LAB_SUMMARY_TEXT = os.environ.get("LAB_SUMMARY", "")
LAB_SUMMARY_MAX_CHARS = int(os.environ.get("LAB_SUMMARY_MAX_CHARS", "200"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("linephone")

# ---------------------------------------------------------------------------
# Frame ring buffer with position tracking
# ---------------------------------------------------------------------------


class _Frame:
    __slots__ = ("pos", "sender_id", "wire")

    def __init__(self, pos: int, sender_id: str, wire: bytes):
        self.pos = pos
        self.sender_id = sender_id
        self.wire = wire


class _RingBuffer:
    """Thread-safe ring buffer of Opus wire-frames with monotonic position IDs."""

    def __init__(self, capacity: int):
        self._cap = capacity
        self._buf: list[_Frame] = []
        self._lock = threading.Lock()
        self._next_pos: int = 1  # 1-based; 0 means "nothing yet"

    @property
    def head(self) -> int:
        """Position of the latest frame (0 if empty)."""
        with self._lock:
            return self._next_pos - 1

    def append(self, sender_id: str, wire: bytes) -> int:
        """Append a frame. Returns assigned position."""
        with self._lock:
            pos = self._next_pos
            self._next_pos += 1
            self._buf.append(_Frame(pos, sender_id, wire))
            if len(self._buf) > self._cap:
                self._buf = self._buf[len(self._buf) - self._cap:]
        return pos

    def read_from(self, start: int, device_id: Optional[str],
                  echo: bool) -> list[_Frame]:
        """Return frames with pos > start.  Filters out sender_id == device_id
        unless echo is True."""
        with self._lock:
            if not self._buf:
                return []
            first_pos = self._buf[0].pos
            if start < first_pos:
                idx = 0
            else:
                idx = start - first_pos + 1
            if idx >= len(self._buf):
                return []
            frames = self._buf[idx:]
        if device_id and not echo:
            frames = [f for f in frames if f.sender_id != device_id]
        return frames



_ring = _RingBuffer(RING_CAPACITY)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Opus Linephone Relay")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# /api/device/linephone/
# ---------------------------------------------------------------------------

@app.get("/api/device/linephone/")
async def linephone_get(
    start: int = Query(0, description="Position to resume from (0 = latest)"),
    device_id: str = Query("", description="Caller device id (to skip own frames)"),
    echo: int = Query(0, description="1 = echo mode (include own frames)"),
):
    """GET: server -> device.  Opus stream from a given position.

    Wire format: [BE16 length][opus packet bytes] repeated per frame.

    Response header X-Linephone-Head contains the current head position
    at the time the stream starts.
    """
    is_echo = echo == 1
    dev = device_id or None

    frames = _ring.read_from(start, dev, is_echo)
    head = frames[-1].pos if frames else _ring.head

    logger.info("GET: start=%d device_id=%s echo=%s -> %d frames, head=%d",
                start, dev or "-", is_echo, len(frames), head)

    async def relay():
        for f in frames:
            yield f.wire

    return StreamingResponse(
        relay(),
        media_type="audio/opus",
        headers={
            "X-Audio-Format": "opus_be16len",
            "X-Opus-Sample-Rate": str(OPUS_SR),
            "X-Opus-Frame-Ms": str(OPUS_FRAME_MS),
            "X-Opus-Bitrate": str(OPUS_BITRATE),
            "X-Opus-Channels": str(OPUS_CHANNELS),
            "X-Linephone-Head": str(head),
            "X-Linephone-Count": str(len(frames)),
            "Cache-Control": "no-cache, no-store",
        },
    )


@app.post("/api/device/linephone/")
async def linephone_post(request: Request):
    """POST: device -> server.  Receive Opus frames, store & broadcast.

    Body: raw binary [BE16 length][opus packet]... frames,
          or application/json {"audio": "<base64 of framed opus>"}.

    Header X-Device-Id or JSON field "device_id" identifies the sender.

    Response:
        {"ok": true, "frames": N, "head": <latest position>}
    """
    content_type = request.headers.get("content-type", "")
    if "json" in content_type:
        body = await request.json()
        b64 = body.get("audio", "")
        if not b64:
            raise HTTPException(400, "audio field required")
        try:
            raw_data = base64.b64decode(b64)
        except Exception:
            raise HTTPException(400, "invalid base64")
        sender = body.get("device_id") or request.headers.get("x-device-id", "")
    else:
        raw_data = await request.body()
        if not raw_data:
            raise HTTPException(400, "empty body")
        sender = request.headers.get("x-device-id", "")

    sender = sender.strip() or "anonymous"

    # Parse [BE16 length][opus packet]... and store each frame
    pos = 0
    frame_count = 0
    last_pos = 0
    while pos + 2 <= len(raw_data):
        pkt_len = struct.unpack(">H", raw_data[pos:pos + 2])[0]
        pos += 2
        if pos + pkt_len > len(raw_data):
            break
        wire = raw_data[pos - 2:pos + pkt_len]  # include BE16 header
        pos += pkt_len
        last_pos = _ring.append(sender, wire)
        frame_count += 1

    if frame_count == 0:
        raise HTTPException(400, "no valid opus frames found")

    # Debug: log packet size stats
    if DEBUG_DUMP_DIR:
        pkt_sizes: list[int] = []
        p = 0
        while p + 2 <= len(raw_data):
            plen = struct.unpack(">H", raw_data[p:p + 2])[0]
            p += 2
            if p + plen > len(raw_data):
                break
            pkt_sizes.append(plen)
            p += plen
        logger.info("DEBUG: %d packets, sizes min=%d max=%d avg=%.0f, total=%d bytes",
                     len(pkt_sizes),
                     min(pkt_sizes) if pkt_sizes else 0,
                     max(pkt_sizes) if pkt_sizes else 0,
                     sum(pkt_sizes) / len(pkt_sizes) if pkt_sizes else 0,
                     len(raw_data))

    # Debug: decode Opus -> WAV (s16le) for ffplay ./dump/0000.wav
    if DEBUG_DUMP_DIR:
        global _dump_counter
        import wave
        os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
        path = os.path.join(DEBUG_DUMP_DIR, f"{_dump_counter:04d}.wav")
        try:
            import opuslib
            dec = opuslib.Decoder(OPUS_SR, OPUS_CHANNELS)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(OPUS_CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(OPUS_SR)
                p = 0
                while p + 2 <= len(raw_data):
                    plen = struct.unpack(">H", raw_data[p:p + 2])[0]
                    p += 2
                    if p + plen > len(raw_data):
                        break
                    wf.writeframes(dec.decode(raw_data[p:p + plen], OPUS_FRAME_SAMPLES))
                    p += plen
            logger.info("DEBUG: dumped %d frames to %s", frame_count, path)
        except Exception as e:
            logger.warning("DEBUG: dump failed: %s", e)
        _dump_counter += 1

    logger.info("POST: %s sent %d frames, head=%d", sender, frame_count, last_pos)
    return {"ok": True, "frames": frame_count, "head": last_pos}


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@app.get("/api/lab/summary")
async def get_lab_summary():
    """Lab self-description for proxy fan-out / GUI trust-ranking."""
    import datetime
    tagline = LAB_SUMMARY_TEXT.strip()
    if not tagline:
        tagline = f"Opus linephone relay (sr={OPUS_SR}, ch={OPUS_CHANNELS})"
    if len(tagline) > LAB_SUMMARY_MAX_CHARS:
        tagline = tagline[:LAB_SUMMARY_MAX_CHARS - 1] + "..."
    return {
        "lab_id": LAB_ID,
        "name": LAB_NAME,
        "role": "linephone",
        "summary": tagline,
        "trust": {
            "corpus_chunks": 0,
            "files": 0,
            "papers": 0,
            "by_type": {},
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "embed_model": None,
            "embed_dim": None,
        },
        "current_meta": {
            "opus_bitrate": OPUS_BITRATE,
            "sample_rate": OPUS_SR,
            "head": _ring.head,
        },
        "rebuilding": False,
        "facr_active": False,
        "recent_additions": [],
    }


@app.get("/api/qa/timeline")
async def qa_timeline(
    start: int = 0,
    end: int = -1,
    limit: int = 10,
    publish: int = 0,
):
    if publish:
        return {"count": 0, "max_publish_id": 0, "items": []}
    return {"count": 0, "max_qa_id": 0, "items": []}


@app.get("/api/config")
async def api_config():
    return {
        "linephone_path": "/api/device/linephone/",
        "opus_sr": OPUS_SR,
        "opus_channels": OPUS_CHANNELS,
        "opus_frame_ms": OPUS_FRAME_MS,
        "opus_bitrate": OPUS_BITRATE,
        "ring_capacity": RING_CAPACITY,
        "head": _ring.head,
    }


_aws_creds: dict | None = None


def set_aws_credentials(access_key_id: str, secret_access_key: str,
                        session_token: str | None,
                        expiration: str | None) -> None:
    global _aws_creds
    _aws_creds = {
        "access_key_id": access_key_id,
        "secret_access_key": secret_access_key,
        "session_token": session_token,
        "expiration": expiration,
    }


@app.post("/api/internal/aws-credentials")
async def internal_set_aws_credentials(request: Request):
    if not INTERNAL_TOKEN or request.headers.get("x-internal-token") != INTERNAL_TOKEN:
        raise HTTPException(403, "forbidden")
    body = await request.json()
    akid = body.get("access_key_id")
    secret = body.get("secret_access_key")
    if not akid or not secret:
        raise HTTPException(400, "access_key_id / secret_access_key required")
    set_aws_credentials(
        access_key_id=akid,
        secret_access_key=secret,
        session_token=body.get("session_token"),
        expiration=body.get("expiration"),
    )
    logger.info("aws creds installed (akid=%s..., exp=%s)",
                akid[:6], body.get("expiration"))
    return {"ok": True, "expiration": body.get("expiration")}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Opus linephone relay")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8891)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
