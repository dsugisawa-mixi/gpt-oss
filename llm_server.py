"""
SFX prompt LLM server, powered by gpt-oss-20b (vLLM backend).

This server takes a short free-form sound description in ANY language
(typically Japanese, e.g. "大きな犬") and uses the LLM to rewrite it into a
single, rich, ENGLISH Stable Audio 3 sound-effect prompt, e.g.:

    input : "大きな犬"
    output: "A large dog barking loudly, realistic animal sound effect,
             close distance, outdoor environment, clean recording,
             high quality SFX"

It can also forward that generated prompt straight to the Stable Audio 3
generator (run separately with `uv run python serve_api.py --model medium
--port 8000`) and stream the resulting audio back, so a client can go from
a Japanese word to a rendered sound effect in one request.

This module is a stripped-down sibling of your_professor_server.py — it
reuses only the proven LLM-serving machinery (vLLM TokenGenerator + Harmony
encoding + the streaming generate loop) and drops all of the presentation /
auth / RAG / upload platform code, which is irrelevant to SFX prompting.

Usage:
    # 1. Start the SFX audio generator (separate repo/process):
    #    cd /home/ubuntu/git/stable-audio-3
    #    uv run python serve_api.py --model medium --port 8000
    #
    # 2. Start this LLM prompt server:
    python llm_server.py [--checkpoint openai/gpt-oss-20b] [--port 8081] \
        [--sfx-backend http://127.0.0.1:8000]

Endpoints:
    GET  /health           — liveness + which SFX backend is configured
    POST /generate_prompt  — {text} -> {prompt}  (LLM rewrite only, no audio)
    POST /generate         — {text, duration, steps, ...} -> audio/opus
                             (LLM rewrite, then proxy to the SFX generator;
                              the generated prompt is echoed in the
                              X-SFX-Prompt response header)
"""

import argparse
import asyncio
import base64
import datetime
import logging
import os
import re
import struct
import threading
import time
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import httpx
import torch  # noqa: F401  (kept so vLLM picks up CUDA env consistently)
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    load_harmony_encoding,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sfx-llm")

# --- Runtime config (overridable by CLI / env) ---
# Where the Stable Audio 3 generator (serve_api.py) listens. The /generate
# endpoint POSTs the LLM-written prompt here and streams the audio back.
SFX_BACKEND = os.environ.get("SFX_BACKEND", "http://127.0.0.1:8000")
SFX_GENERATE_PATH = os.environ.get("SFX_GENERATE_PATH", "/generate")
# Audio generation on the SFX side is GPU-heavy and can take a while; keep a
# generous read timeout so short clips don't get cut off.
SFX_TIMEOUT_S = float(os.environ.get("SFX_TIMEOUT_S", "120"))

# Lab self-description, surfaced at /api/lab/summary for proxy fan-out /
# GUI trust-ranking (same contract as the linephone relay in tranceiver.py).
LAB_ID = os.environ.get("LAB_ID", "lab-sfx-llm").strip()
LAB_NAME = os.environ.get("LAB_NAME", "SFX Prompt LLM (gpt-oss-20b)")
LAB_SUMMARY_TEXT = os.environ.get("LAB_SUMMARY", "")
LAB_SUMMARY_MAX_CHARS = int(os.environ.get("LAB_SUMMARY_MAX_CHARS", "200"))

# --- Linephone ring + Voice -> SFX pipeline ----------------------------------
# This server is itself a linephone node (same wire contract as tranceiver.py):
#   POST /api/device/linephone/  -> append framed Opus into the ring
#   GET  /api/device/linephone/  -> poll framed Opus back out of the ring
# On top of the relay, an incoming voice batch is transcribed (Whisper), turned
# into an English SFX prompt (LLM) and rendered to audio (Stable Audio 3); the
# resulting framed Opus is injected back into the SAME ring as SFX_SENDER so GET
# pollers play it. Heavy deps (numpy / opuslib / faster-whisper) are imported
# lazily so a node without them can still serve /generate_prompt.
OPUS_SR = int(os.environ.get("OPUS_SR", "16000"))
OPUS_CHANNELS = int(os.environ.get("OPUS_CHANNELS", "1"))
OPUS_FRAME_MS = int(os.environ.get("OPUS_FRAME_MS", "20"))
OPUS_FRAME_SAMPLES = OPUS_SR * OPUS_FRAME_MS // 1000
OPUS_BITRATE = int(os.environ.get("OPUS_BITRATE", "16000"))
RING_CAPACITY = int(os.environ.get("RING_CAPACITY", "50000"))  # ~16 min at 20ms

# Voice -> SFX pipeline knobs.
VOICE2SFX_ENABLE = os.environ.get("VOICE2SFX_ENABLE", "1") == "1"
# Skip transcription for batches shorter than this (avoids whispering silence).
VOICE2SFX_MIN_MS = int(os.environ.get("VOICE2SFX_MIN_MS", "300"))
# SFX clip length for the voice pipeline (seconds).
VOICE_SFX_DURATION = float(os.environ.get("VOICE_SFX_DURATION", "5"))
# sender_id stamped on injected SFX frames (kept distinct from any caller id so
# the GET filter never confuses generated audio with a live talker, and so the
# POST handler never re-feeds generated audio into the pipeline).
SFX_SENDER = os.environ.get("SFX_SENDER", "sfx")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")
WHISPER_LANG = (os.environ.get("WHISPER_LANG", "ja").strip() or None)


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
                  echo: bool) -> list:
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

_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper():
    """Lazily load the faster-whisper model (first voice POST pays the cost)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info("loading Whisper model %r on %s (%s) ...",
                    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
        _whisper_model = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        logger.info("Whisper model loaded.")
    return _whisper_model


def _iter_opus_packets(raw_data: bytes):
    """Yield the opus payload of each [BE16 length][opus packet] wire-frame."""
    pos = 0
    n = len(raw_data)
    while pos + 2 <= n:
        pkt_len = struct.unpack(">H", raw_data[pos:pos + 2])[0]
        pos += 2
        if pos + pkt_len > n:
            break
        yield raw_data[pos:pos + pkt_len]
        pos += pkt_len


def _split_wire_frames(framed: bytes) -> list:
    """Split framed Opus into [BE16 length][opus] wire frames (header kept,
    ready to hand to _ring.append)."""
    out = []
    pos = 0
    n = len(framed)
    while pos + 2 <= n:
        pkt_len = struct.unpack(">H", framed[pos:pos + 2])[0]
        end = pos + 2 + pkt_len
        if end > n:
            break
        out.append(framed[pos:end])
        pos = end
    return out


def _decode_to_pcm_f32(raw_data: bytes):
    """Decode framed Opus -> mono float32 PCM in [-1, 1] at OPUS_SR."""
    import numpy as np
    import opuslib

    dec = opuslib.Decoder(OPUS_SR, OPUS_CHANNELS)
    chunks = []
    for pkt in _iter_opus_packets(raw_data):
        pcm = dec.decode(pkt, OPUS_FRAME_SAMPLES)  # int16 LE bytes
        chunks.append(np.frombuffer(pcm, dtype=np.int16))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
    if OPUS_CHANNELS > 1:
        pcm = pcm.reshape(-1, OPUS_CHANNELS).mean(axis=1)
    return pcm


def _transcribe(pcm) -> str:
    """Whisper transcription of mono 16 kHz float32 PCM -> text (blocking)."""
    model = _get_whisper()
    # faster-whisper's CTranslate2 model isn't safe to call concurrently.
    with _whisper_lock:
        segments, _info = model.transcribe(pcm, language=WHISPER_LANG)
        return "".join(seg.text for seg in segments).strip()


async def _voice_to_sfx(raw_data: bytes, sender: str) -> None:
    """Background task: voice frames -> text -> SFX -> inject framed Opus into
    the ring as SFX_SENDER so GET pollers play it."""
    try:
        n_frames = sum(1 for _ in _iter_opus_packets(raw_data))
        if n_frames * OPUS_FRAME_MS < VOICE2SFX_MIN_MS:
            return  # too short to be an utterance; skip

        pcm = await asyncio.to_thread(_decode_to_pcm_f32, raw_data)
        if pcm.size == 0:
            return
        text = await asyncio.to_thread(_transcribe, pcm)
        if not text:
            logger.info("voice2sfx: %s -> (empty transcription), skip", sender)
            return
        logger.info("voice2sfx: %s -> transcript %r", sender, text)

        sfx_prompt = build_sfx_prompt(text)
        logger.info("voice2sfx: %r -> %r", text, sfx_prompt)

        audio, _media_type = await _call_sfx_backend(
            sfx_prompt,
            duration=VOICE_SFX_DURATION,
            steps=8,
            cfg_scale=1.0,
            seed=-1,
            mono=True,
            log_tag="voice2sfx",
        )

        wire_frames = _split_wire_frames(audio)
        if not wire_frames:
            logger.warning("voice2sfx: SFX response had no frames (%d bytes); "
                           "is the backend returning framed Opus?", len(audio))
            return
        last = 0
        for wire in wire_frames:
            last = _ring.append(SFX_SENDER, wire)
        logger.info("voice2sfx: injected %d SFX frames (prompt=%r), head=%d",
                    len(wire_frames), sfx_prompt, last)
    except Exception as exc:  # never let a background failure crash the loop
        logger.warning("voice2sfx failed: %r", exc)

# vLLM token-generator + harmony encoding, populated in main().
generator = None
encoding = None


# =====================================================================
# System prompt: free-form sound idea -> English Stable Audio 3 SFX prompt
# =====================================================================
SYSTEM_PROMPT = """\
You are a sound-design prompt engineer for the Stable Audio 3 text-to-audio
model. Your job is to turn a short, free-form description of a sound — which
may be written in Japanese or any other language — into ONE rich, vivid
ENGLISH sound-effect prompt that Stable Audio 3 can render well.

Always write the prompt in ENGLISH, even when the input is Japanese.

Build the prompt as a single comma-separated phrase that names the sound and
then qualifies it. Cover, when they make sense:
- the main sound source and what it is doing (e.g. "a large dog barking loudly")
- loudness / intensity (loud, soft, faint, thunderous)
- perspective / distance (close distance, mid distance, distant, far away)
- the acoustic environment (indoor room, outdoor environment, forest, city street, large hall with reverb)
- texture / character (deep, sharp, metallic, wet, crackling, rumbling)
- and finish with recording-quality / type tags such as
  "realistic sound effect", "clean recording", "high quality SFX",
  "field recording", or "foley" as appropriate.

Example:
    input : 大きな犬
    output: A large dog barking loudly, realistic animal sound effect, close distance, outdoor environment, clean recording, high quality SFX

Rules:
- Output ONLY the final English prompt. No translation, no explanation,
  no quotation marks, no markdown, no bullet points, no "Prompt:" prefix.
- One single line. Roughly 8 to 40 words.
- Describe sound only. Never invent music lyrics, speech, or words being said
  unless the input explicitly asks for spoken/sung content.
- Stay faithful to the input: do not change a cat into a dog, or quiet into
  loud. Only add plausible acoustic detail consistent with the input.
"""


def _to_harmony_messages(
    prompt_messages: list[dict],
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
) -> list[Message]:
    """Convert simple role/content dicts to Harmony Message objects.

    LOW reasoning effort by default — SFX prompt rewriting is a short,
    well-specified transformation that does not need long chain-of-thought,
    and lower effort keeps latency down.
    """
    harmony_msgs: list[Message] = []
    first_system = True

    for msg in prompt_messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            if first_system:
                first_system = False
                sys_content = (
                    SystemContent.new()
                    .with_reasoning_effort(reasoning_effort)
                    .with_conversation_start_date(
                        datetime.datetime.now().strftime("%Y-%m-%d")
                    )
                    .with_required_channels(["analysis", "final"])
                )
                harmony_msgs.append(
                    Message.from_role_and_content(Role.SYSTEM, sys_content)
                )
                dev_content = DeveloperContent.new().with_instructions(content)
                harmony_msgs.append(
                    Message.from_role_and_content(Role.DEVELOPER, dev_content)
                )
            else:
                dev_content = DeveloperContent.new().with_instructions(content)
                harmony_msgs.append(
                    Message.from_role_and_content(Role.DEVELOPER, dev_content)
                )
        elif role == "user":
            harmony_msgs.append(Message.from_role_and_content(Role.USER, content))
        elif role == "assistant":
            harmony_msgs.append(
                Message.from_role_and_content(Role.ASSISTANT, content).with_channel(
                    "final"
                )
            )

    return harmony_msgs


def generate_reply(
    prompt_messages: list[dict],
    max_new_tokens: int = 512,
    max_reply_tokens: int = 160,
    temperature: float = 0.7,
) -> tuple[str, bool]:
    """Run one forward pass and return (final_reply_text, truncated)."""
    t0 = time.perf_counter()

    harmony_msgs = _to_harmony_messages(prompt_messages)
    conversation = Conversation.from_messages(harmony_msgs)
    tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    # Force the model to start in the analysis channel; otherwise it mimics
    # final-channel-only history and skips reasoning entirely.
    analysis_prefill = encoding.encode(
        "<|channel|>analysis<|message|>", allowed_special="all"
    )
    tokens = tokens + analysis_prefill
    stop_tokens = encoding.stop_tokens_for_assistant_actions()

    t1 = time.perf_counter()

    parser = StreamableParser(encoding, role=Role.ASSISTANT)
    # Replay the prefill so the parser knows we've opened the analysis channel.
    for t in analysis_prefill:
        parser.process(t)

    final_parts: list[str] = []
    channels_seen: set[str] = set()
    gen_count = 0
    final_token_count = 0
    truncated = False
    runaway = False

    # Steady-state on this GPU is ~50-65 tok/s; anything above 2000 tok/s
    # means vLLM is yielding without doing real forward passes (corrupted
    # KV/prefix cache). Check after a short warm-up so we don't false-positive
    # on legitimate short replies.
    RUNAWAY_TOK_PER_SEC = 2000.0
    RUNAWAY_WARMUP_TOKENS = 32

    for predicted_token in generator.generate(
        tokens,
        stop_tokens=stop_tokens,
        temperature=temperature,
        max_tokens=max_new_tokens,
    ):
        parser.process(predicted_token)
        gen_count += 1
        if gen_count == RUNAWAY_WARMUP_TOKENS:
            elapsed_s = time.perf_counter() - t1
            if elapsed_s > 0 and gen_count / elapsed_s > RUNAWAY_TOK_PER_SEC:
                runaway = True
                logger.error(
                    "generate: runaway detected (%d tok in %.1fms, %.0f tok/s) "
                    "— vLLM KV/prefix cache state appears corrupted; aborting "
                    "this reply (recommend restarting the LLM service)",
                    gen_count, elapsed_s * 1000, gen_count / elapsed_s,
                )
                break
        ch = parser.current_channel
        if ch:
            channels_seen.add(ch)
        delta = parser.last_content_delta
        if not delta:
            continue
        if ch == "final":
            final_parts.append(delta)
            final_token_count += 1
            if final_token_count >= max_reply_tokens:
                truncated = True
                break

    t2 = time.perf_counter()
    gen_ms = (t2 - t1) * 1000
    tok_per_sec = gen_count / (t2 - t1) if (t2 - t1) > 0 else 0

    reply = "".join(final_parts)
    if runaway:
        reply = ""
        truncated = True
    logger.info(
        "generate: reply_len=%d truncated=%s in=%d out=%d (final=%d) "
        "encode=%.1fms gen=%.1fms (%.1f tok/s) channels=%s%s",
        len(reply), truncated, len(tokens), gen_count, final_token_count,
        (t1 - t0) * 1000, gen_ms, tok_per_sec, channels_seen,
        " RUNAWAY" if runaway else "",
    )
    return reply, truncated


# Matches a leading label like `Prompt:` / `Output:` the model sometimes adds.
_LABEL_RE = re.compile(r"^\s*(prompt|output|sfx|answer)\s*[:：]\s*", re.IGNORECASE)


def _clean_prompt(text: str) -> str:
    """Normalise the LLM output into a single clean SFX prompt line."""
    text = (text or "").strip()
    if not text:
        return ""
    # Take the first non-empty line — guard against the model adding notes
    # on subsequent lines despite the instructions.
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    text = _LABEL_RE.sub("", text)
    # Strip wrapping quotes / backticks.
    text = text.strip().strip("`")
    if len(text) >= 2 and text[0] in "\"'“”" and text[-1] in "\"'“”":
        text = text[1:-1].strip()
    return text


def build_sfx_prompt(text: str) -> str:
    """Rewrite a free-form (possibly Japanese) sound idea into an English
    Stable Audio 3 SFX prompt. Falls back to the cleaned raw input if the
    model returns nothing usable."""
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text.strip()},
    ]
    reply, _truncated = generate_reply(prompt_messages)
    cleaned = _clean_prompt(reply)
    if not cleaned:
        # Graceful degrade: pass the raw description through with a generic
        # SFX tail so the audio backend still gets something usable.
        logger.warning("build_sfx_prompt: empty/blank LLM reply for %r — falling back", text)
        cleaned = f"{text.strip()}, sound effect, clean recording, high quality SFX"
    return cleaned


# =====================================================================
# FastAPI app
# =====================================================================
app = FastAPI(title="SFX Prompt LLM Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    text: str = Field(..., description="Free-form sound description (any language)")


class PromptResponse(BaseModel):
    input: str
    prompt: str


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Free-form sound description (any language)")
    # SFX generator passthrough knobs (forwarded to serve_api.py /generate).
    duration: float = 10.0
    steps: int = 8
    cfg_scale: float = 1.0
    seed: int = -1
    negative_prompt: Optional[str] = None
    mono: bool = False


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": generator is not None,
        "sfx_backend": SFX_BACKEND.rstrip("/") + SFX_GENERATE_PATH,
    }


@app.get("/api/lab/summary")
async def get_lab_summary():
    """Lab self-description for proxy fan-out / GUI trust-ranking."""
    tagline = LAB_SUMMARY_TEXT.strip()
    if not tagline:
        tagline = "Free-form (JP/any) sound idea -> English Stable Audio 3 SFX prompt"
    if len(tagline) > LAB_SUMMARY_MAX_CHARS:
        tagline = tagline[:LAB_SUMMARY_MAX_CHARS - 1] + "..."
    return {
        "lab_id": LAB_ID,
        "name": LAB_NAME,
        "role": "sfx-llm",
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
            "model_loaded": generator is not None,
            "sfx_backend": SFX_BACKEND.rstrip("/") + SFX_GENERATE_PATH,
            "opus_bitrate": OPUS_BITRATE,
            "sample_rate": OPUS_SR,
            "head": _ring.head,
        },
        "rebuilding": False,
        "facr_active": False,
        "recent_additions": [],
    }


@app.post("/generate_prompt", response_model=PromptResponse)
async def generate_prompt(req: PromptRequest):
    """LLM rewrite only: free-form text -> English SFX prompt. No audio."""
    if generator is None:
        raise HTTPException(503, "model not loaded")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "`text` is required")
    prompt = build_sfx_prompt(text)
    logger.info("generate_prompt: %r -> %r", text, prompt)
    return PromptResponse(input=text, prompt=prompt)


async def _call_sfx_backend(
    sfx_prompt: str,
    *,
    duration: float,
    steps: int,
    cfg_scale: float,
    seed: int,
    mono: bool,
    negative_prompt: Optional[str] = None,
    log_tag: str = "generate",
) -> tuple[bytes, str]:
    """POST an English SFX prompt to the Stable Audio 3 backend and return the
    raw audio bytes + media type. Shared by /generate (which streams it) and
    the voice pipeline (which splits it into ring frames)."""
    payload = {
        "prompt": sfx_prompt,
        "duration": duration,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "mono": mono,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    url = SFX_BACKEND.rstrip("/") + SFX_GENERATE_PATH
    try:
        upstream = await httpx.AsyncClient(timeout=SFX_TIMEOUT_S).post(
            url, json=payload
        )
    except httpx.HTTPError as exc:
        logger.error("%s: SFX backend request failed: %s", log_tag, exc)
        raise HTTPException(502, f"SFX backend at {url} unreachable: {exc}")

    if upstream.status_code != 200:
        detail = upstream.text[:500]
        logger.error("%s: SFX backend %d: %s", log_tag, upstream.status_code, detail)
        raise HTTPException(502, f"SFX backend error {upstream.status_code}: {detail}")

    media_type = upstream.headers.get("content-type", "audio/opus")
    return upstream.content, media_type


async def _synthesize_sfx_response(
    sfx_prompt: str,
    *,
    duration: float,
    steps: int,
    cfg_scale: float,
    seed: int,
    mono: bool,
    negative_prompt: Optional[str] = None,
    log_tag: str = "generate",
) -> StreamingResponse:
    """Render an SFX prompt and stream the audio back to an HTTP caller."""
    audio, media_type = await _call_sfx_backend(
        sfx_prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        mono=mono,
        negative_prompt=negative_prompt,
        log_tag=log_tag,
    )

    def _stream():
        chunk = 64 * 1024
        for i in range(0, len(audio), chunk):
            yield audio[i : i + chunk]

    # Header values must be latin-1 safe; the prompt is ASCII English so this
    # is fine, but guard against stray non-ascii just in case.
    safe_prompt = sfx_prompt.encode("ascii", "ignore").decode("ascii")
    return StreamingResponse(
        _stream(),
        media_type=media_type,
        headers={
            "Content-Length": str(len(audio)),
            "Content-Disposition": 'inline; filename="sfx.opus"',
            "X-SFX-Prompt": safe_prompt,
        },
    )


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Full pipeline: free-form text -> English SFX prompt (LLM) -> audio.

    The generated English prompt is forwarded to the Stable Audio 3 backend
    (serve_api.py) and the resulting audio is streamed back to the caller.
    The prompt that was actually used is returned in the X-SFX-Prompt header.
    """
    if generator is None:
        raise HTTPException(503, "model not loaded")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "`text` is required")

    sfx_prompt = build_sfx_prompt(text)
    logger.info("generate: %r -> %r", text, sfx_prompt)

    return await _synthesize_sfx_response(
        sfx_prompt,
        duration=req.duration,
        steps=req.steps,
        cfg_scale=req.cfg_scale,
        seed=req.seed,
        mono=req.mono,
        negative_prompt=req.negative_prompt,
        log_tag="generate",
    )


@app.get("/api/device/linephone/")
async def linephone_get(
    start: int = Query(0, description="Position to resume from (0 = latest)"),
    device_id: str = Query("", description="Caller device id (to skip own frames)"),
    echo: int = Query(0, description="1 = echo mode (include own frames)"),
):
    """GET: server -> device.  Framed Opus stream from a given position.

    Wire format: [BE16 length][opus packet bytes] repeated per frame. The SFX
    audio synthesised from voice POSTs (sender=SFX_SENDER) shows up here.
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
    """POST: device -> server.  Receive framed Opus voice, store it in the ring,
    and (unless it is our own SFX audio) kick off the voice -> SFX pipeline,
    which injects the synthesised audio back into the ring for GET pollers.

    Body: raw binary [BE16 length][opus packet]... frames,
          or application/json {"audio": "<base64 of framed opus>"}.
    Sender id comes from header X-Device-Id or JSON field "device_id".
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

    frame_count = 0
    last_pos = 0
    for wire in _split_wire_frames(raw_data):
        last_pos = _ring.append(sender, wire)
        frame_count += 1

    if frame_count == 0:
        raise HTTPException(400, "no valid opus frames found")

    # Voice -> SFX: transcribe this batch and synthesise a sound effect,
    # injecting it back into the ring for GET pollers. Fire-and-forget so the
    # device's POST returns immediately; SFX_SENDER frames are never re-fed.
    if VOICE2SFX_ENABLE and sender != SFX_SENDER:
        asyncio.create_task(_voice_to_sfx(raw_data, sender))

    logger.info("POST: %s sent %d frames, head=%d", sender, frame_count, last_pos)
    return {"ok": True, "frames": frame_count, "head": last_pos}


def main():
    global generator, encoding, SFX_BACKEND

    parser = argparse.ArgumentParser(description="SFX Prompt LLM HTTP Server")
    parser.add_argument("--checkpoint", default="openai/gpt-oss-20b", help="Model name or path")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--context-length", type=int, default=16384,
                        help="Max model context length passed to the vLLM engine")
    parser.add_argument("--sfx-backend", default=SFX_BACKEND,
                        help="Base URL of the Stable Audio 3 generator (serve_api.py)")
    args = parser.parse_args()

    SFX_BACKEND = args.sfx_backend

    from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator

    print(f"Loading model: {args.checkpoint} (vLLM backend, max_model_len={args.context_length}) ...")
    generator = VLLMGenerator(
        args.checkpoint,
        tensor_parallel_size=1,
        max_model_len=args.context_length,
    )
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    print("Model loaded with vLLM backend.")
    print(f"SFX backend: {SFX_BACKEND.rstrip('/') + SFX_GENERATE_PATH}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
