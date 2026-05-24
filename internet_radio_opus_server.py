"""
Internet-radio -> Opus re-encoder streaming server.

On startup, fetches the full Japan station list from radio-browser.info and
begins a background loop that plays each station for ~3 minutes, cycling
through every station endlessly.

Clients connect to ``POST /api/tts/generate_stream`` and receive the live
Opus stream (``[BE16 length][opus packet]`` framing, same as server-design.py).
Multiple clients share the same underlying radio -> Opus pipeline.

Usage:
    pip install fastapi uvicorn httpx miniaudio opuslib
    # libopus must be installed (brew install opus / apt install libopus-dev)

    python internet_radio_opus_server.py [--port 8891]

Endpoints:
    POST /api/tts/generate_stream  -- join the live Opus stream
    GET  /api/lab/summary          -- lab self-description for proxy registry
    GET  /api/config               -- returns public TTS path
"""

import argparse
import asyncio
import ctypes
import datetime
import json
import logging
import os
import struct
import threading
import time
from typing import Optional

import httpx
import miniaudio

# opuslib needs libopus on the library path.  On macOS with Homebrew:
#   export DYLD_LIBRARY_PATH=/opt/homebrew/lib
_opus_lib_hint = "/opt/homebrew/lib"
if os.path.isdir(_opus_lib_hint):
    _cur = os.environ.get("DYLD_LIBRARY_PATH", "")
    if _opus_lib_hint not in _cur:
        os.environ["DYLD_LIBRARY_PATH"] = f"{_opus_lib_hint}:{_cur}" if _cur else _opus_lib_hint

import opuslib  # noqa: E402  (after env tweak)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RADIO_BROWSER_API = os.environ.get(
    "RADIO_BROWSER_API", "https://de1.api.radio-browser.info"
)

# Lab identity for proxy registration (same contract as your_professor_server)
LAB_ID = os.environ.get("LAB_ID", "lab-radio").strip()
LAB_NAME = os.environ.get("LAB_NAME", "Internet Radio (Opus)")
LAB_SUMMARY_TEXT = os.environ.get("LAB_SUMMARY", "")
LAB_SUMMARY_MAX_CHARS = int(os.environ.get("LAB_SUMMARY_MAX_CHARS", "200"))

# Opus streaming parameters -- same scheme as server-design.py.
# Framing: each packet is written as [BE16 length][opus packet bytes].
OPUS_SR = int(os.environ.get("OPUS_SR", "16000"))
OPUS_CHANNELS = int(os.environ.get("OPUS_CHANNELS", "1"))
OPUS_FRAME_MS = int(os.environ.get("OPUS_FRAME_MS", "20"))
OPUS_FRAME_SAMPLES = OPUS_SR * OPUS_FRAME_MS // 1000  # 480
OPUS_BITRATE = int(os.environ.get("OPUS_BITRATE", "16000"))  # bps

# How long to stay on each station before switching (seconds).
STATION_SWITCH_INTERVAL = int(os.environ.get("STATION_SWITCH_INTERVAL", "180"))

INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "")

TTS_PUBLIC_PATH = "/api/tts/generate_stream"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("radio-opus")

# ---------------------------------------------------------------------------
# Shared broadcast state
# ---------------------------------------------------------------------------
# The background loop produces Opus wire-frames and fans them out to all
# connected clients via per-client asyncio.Queue instances.

_clients: list[asyncio.Queue] = []       # active client queues
_clients_lock = threading.Lock()

_current_station: dict = {}               # currently playing station info
_next_switch_at: float = 0.0              # time.time() of next station switch
_stations_list: list[dict] = []           # full JP station list


def _broadcast(data: bytes):
    """Push a wire-frame to every connected client (non-blocking, drop on full)."""
    with _clients_lock:
        for q in _clients:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass  # slow client, drop frame


def _add_client() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _clients_lock:
        _clients.append(q)
    return q


def _remove_client(q: asyncio.Queue):
    with _clients_lock:
        try:
            _clients.remove(q)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Background station-loop worker (runs in a daemon thread)
# ---------------------------------------------------------------------------

def _fetch_stations_sync() -> list[dict]:
    """Fetch JP stations from radio-browser.info (synchronous)."""
    url = f"{RADIO_BROWSER_API.rstrip('/')}/json/stations/bycountry/Japan"
    params = {
        "limit": "500",
        "order": "votes",
        "reverse": "true",
        "hidebroken": "true",
    }
    with httpx.Client(timeout=15) as client:
        resp = client.get(url, params=params,
                          headers={"User-Agent": "radio-opus-server/1.0"})
        resp.raise_for_status()
    raw = resp.json()
    stations = []
    for s in raw:
        stream_url = (s.get("url_resolved") or s.get("url") or "").strip()
        if not stream_url:
            continue
        if s.get("hls") == 1 or ".m3u8" in stream_url:
            continue
        stations.append({
            "stationuuid": s.get("stationuuid"),
            "name": (s.get("name") or "").strip(),
            "url": stream_url,
            "codec": s.get("codec"),
            "bitrate": s.get("bitrate"),
            "country": s.get("country"),
            "countrycode": s.get("countrycode"),
            "tags": s.get("tags"),
            "votes": s.get("votes"),
            "favicon": s.get("favicon"),
        })
    return stations


def _transcode_one_station(station: dict, stop_event: threading.Event):
    """Connect to one station, transcode to Opus, broadcast until stopped."""
    global _current_station

    url = station["url"]
    _current_station = station

    logger.info(">> Now playing: %s  (%s)  url=%s",
                station["name"], station.get("codec"), url)

    ice = None
    try:
        ice = miniaudio.IceCastClient(url)
    except Exception:
        logger.warning("Failed to connect to %s (%s), skipping",
                       station["name"], url)
        return

    def _on_title(_client, title):
        logger.info("Stream title [%s]: %s", station["name"], title)

    ice.update_stream_title = _on_title

    try:
        encoder = opuslib.Encoder(OPUS_SR, OPUS_CHANNELS, opuslib.APPLICATION_AUDIO)
        # opuslib の encoder.* setter は opus_encoder_ctl (variadic) を
        # argtypes 未設定で呼ぶため、ARM64 macOS では値がレジスタ→
        # スタックの ABI 不一致で libopus に届かない。ctypes を直接使い
        # fixed args の argtypes を宣言して variadic 引数を正しく渡す。
        _ctl = ctypes.cdll.LoadLibrary(opuslib.api.libopus._name).opus_encoder_ctl
        _ctl.restype = ctypes.c_int
        _ctl.argtypes = [ctypes.c_void_p, ctypes.c_int]
        # SET_BITRATE (4002)
        _rc = _ctl(encoder.encoder_state, 4002, ctypes.c_int32(OPUS_BITRATE))
        if _rc != 0:
            logger.warning("opus_encoder_ctl SET_BITRATE failed: %d", _rc)
        else:
            logger.info("Opus encoder bitrate set to %d bps (via direct CTL)", OPUS_BITRATE)
        # SET_SIGNAL (4024) = SIGNAL_AUTO (-1000)
        _rc = _ctl(encoder.encoder_state, 4024, ctypes.c_int32(-1000))
        if _rc != 0:
            logger.warning("opus_encoder_ctl SET_SIGNAL failed: %d", _rc)
        else:
            logger.info("Opus encoder signal set to SIGNAL_AUTO (via direct CTL)")
        # SET_COMPLEXITY (4010) = 10 (最高品質)
        _rc = _ctl(encoder.encoder_state, 4010, ctypes.c_int32(10))
        if _rc != 0:
            logger.warning("opus_encoder_ctl SET_COMPLEXITY failed: %d", _rc)
        # SET_VBR (4006) = 1 (有効)
        _rc = _ctl(encoder.encoder_state, 4006, ctypes.c_int32(1))
        if _rc != 0:
            logger.warning("opus_encoder_ctl SET_VBR failed: %d", _rc)
        # SET_VBR_CONSTRAINT (4020) = 0 (unconstrained)
        _rc = _ctl(encoder.encoder_state, 4020, ctypes.c_int32(0))
        if _rc != 0:
            logger.warning("opus_encoder_ctl SET_VBR_CONSTRAINT failed: %d", _rc)
        logger.info("Opus encoder: complexity=10, VBR=1, VBR_CONSTRAINT=0")

        pcm_gen = miniaudio.stream_any(
            ice,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=OPUS_CHANNELS,
            sample_rate=OPUS_SR,
            frames_to_read=OPUS_FRAME_SAMPLES,
        )

        pcm_buf = bytearray()
        bytes_per_frame = OPUS_FRAME_SAMPLES * OPUS_CHANNELS * 2

        for pcm_chunk in pcm_gen:
            if stop_event.is_set():
                break

            pcm_buf.extend(pcm_chunk.tobytes())

            while len(pcm_buf) >= bytes_per_frame:
                raw = bytes(pcm_buf[:bytes_per_frame])
                del pcm_buf[:bytes_per_frame]

                packet = encoder.encode(raw, OPUS_FRAME_SAMPLES)
                wire = struct.pack(">H", len(packet)) + packet
                _broadcast(wire)

    except Exception:
        if not stop_event.is_set():
            logger.exception("Transcode error for %s", station["name"])
    finally:
        if ice:
            try:
                ice.close()
            except Exception:
                pass


def _station_loop(stop_event: threading.Event):
    """Main loop: fetch stations, cycle through them forever."""
    global _stations_list

    while not stop_event.is_set():
        # (Re-)fetch station list
        try:
            _stations_list = _fetch_stations_sync()
            logger.info("Fetched %d JP stations from radio-browser.info",
                        len(_stations_list))
        except Exception:
            logger.exception("Failed to fetch station list, retrying in 30s")
            stop_event.wait(30)
            continue

        if not _stations_list:
            logger.warning("No stations found, retrying in 30s")
            stop_event.wait(30)
            continue

        # Play the first station forever (reconnect on error)
        station = _stations_list[0]
        while not stop_event.is_set():
            _transcode_one_station(station, stop_event)
            if not stop_event.is_set():
                logger.info("Stream ended for %s, reconnecting in 2s...",
                            station["name"])
                stop_event.wait(2)


_stop_event = threading.Event()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

app = FastAPI(title="Internet Radio -> Opus Streamer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _on_startup():
    """Launch the background station-loop thread on server start."""
    t = threading.Thread(target=_station_loop, args=(_stop_event,), daemon=True)
    t.start()
    logger.info("Station loop thread started (switch every %ds)",
                STATION_SWITCH_INTERVAL)


@app.on_event("shutdown")
async def _on_shutdown():
    _stop_event.set()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

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
    """Receive STS creds relayed from the ECS proxy via tunnel_client.
    Guarded by a shared INTERNAL_TOKEN -- the endpoint never traverses
    the public tunnel/proxy path (only callable on the compose-internal
    network from proxy-tunnel)."""
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
    return {"tts_path": TTS_PUBLIC_PATH}


@app.get("/api/lab/summary")
async def get_lab_summary():
    """Lab self-description for proxy fan-out / GUI trust-ranking."""
    tagline = LAB_SUMMARY_TEXT.strip()
    if not tagline:
        tagline = f"Internet Radio -> Opus relay (sr={OPUS_SR}, ch={OPUS_CHANNELS})"
    if len(tagline) > LAB_SUMMARY_MAX_CHARS:
        tagline = tagline[: LAB_SUMMARY_MAX_CHARS - 1] + "..."

    return {
        "lab_id": LAB_ID,
        "name": LAB_NAME,
        "summary": tagline,
        "trust": {
            "corpus_chunks": 0,
            "files": 0,
            "papers": 0,
            "by_type": {},
            "last_updated": _started_at,
            "embed_model": None,
            "embed_dim": None,
        },
        "current_meta": {
            "now_playing": _current_station.get("name", ""),
            "opus_bitrate": OPUS_BITRATE,
            "sample_rate": OPUS_SR,
            "stations_total": len(_stations_list),
        },
        "rebuilding": False,
        "facr_active": False,
        "recent_additions": [],
    }


@app.post("/api/tts/generate_stream")
async def tts_generate_stream(req: Request):
    """Join the live Opus broadcast stream.

    Wire format (same as server-design.py):
        [BE16 length][opus packet bytes]   -- repeated per frame

    All clients receive the same audio from the currently playing station.
    """
    q = _add_client()
    logger.info("Client joined broadcast (total=%d)", len(_clients))

    async def relay():
        try:
            while True:
                chunk = await q.get()
                if chunk is None:
                    logger.info("Client relay got None sentinel, stopping")
                    break
                yield chunk
        except asyncio.CancelledError:
            pass
        finally:
            _remove_client(q)
            logger.info("Client left broadcast (total=%d)", len(_clients))

    return StreamingResponse(
        relay(),
        media_type="audio/opus",
        headers={
            "X-Audio-Format": "opus_be16len",
            "X-Opus-Sample-Rate": str(OPUS_SR),
            "X-Opus-Frame-Ms": str(OPUS_FRAME_MS),
            "X-Opus-Bitrate": str(OPUS_BITRATE),
            "X-Opus-Channels": str(OPUS_CHANNELS),
            "X-Radio-Source": _current_station.get("url", ""),
            "X-Station-Name": _current_station.get("name", ""),
            "Cache-Control": "no-cache, no-store",
        },
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Internet Radio -> Opus streamer")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--switch-interval", type=int, default=None,
                        help="Seconds per station (default: 180)")
    args = parser.parse_args()

    if args.switch_interval:
        global STATION_SWITCH_INTERVAL
        STATION_SWITCH_INTERVAL = args.switch_interval

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
