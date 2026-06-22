"""Minimal HTTP API for Stable Audio 3.

POST /generate with a `prompt` (plus optional generation params) and receive the
result as 16 kHz raw Opus frames in the linephone wire format -- no Ogg
container. The body is a concatenation of `[BE16 length][opus packet]` frames,
the same payload `/api/device/linephone/` consumes.

Run:
    uv run python serve_api.py --model medium --port 8000

Example (de-frame back into Opus packets):
    res = requests.post("http://localhost:8000/generate",
                        json={"prompt": "lofi hip hop beat", "duration": 10})
    raw, pos, packets = res.content, 0, []
    while pos + 2 <= len(raw):
        n = int.from_bytes(raw[pos:pos + 2], "big"); pos += 2
        packets.append(raw[pos:pos + n]); pos += n
"""

import io
import struct
import sys
import threading

# Silence library warnings unless --verbose, matching run_gradio.py. Must run
# before any ML library imports since most warnings fire at import time.
if "--verbose" not in sys.argv:
    import os as _os

    _os.environ.setdefault("PYTHONWARNINGS", "ignore")
    import warnings as _warnings

    _warnings.filterwarnings("ignore")

import torch
import torchaudio
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from stable_audio_3 import StableAudioModel
from stable_audio_3.verbose import set_verbose

# Opus only accepts these container sample rates.
OUTPUT_SAMPLE_RATE = 16000

app = FastAPI(title="Stable Audio 3 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set in main() before the server starts serving.
MODEL: StableAudioModel | None = None
# Diffusion generation is GPU-bound and not safe to run concurrently; serialize it.
_GEN_LOCK = threading.Lock()


async def _parse_params(request: Request) -> dict:
    """Accept both JSON bodies and form/urlencoded POST params."""
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        return await request.json()
    form = await request.form()
    return {k: form[k] for k in form}


def _encode_ogg_opus(audio: torch.Tensor, in_sr: int, mono: bool) -> bytes:
    """audio: (channels, samples) float32 in [-1, 1] -> Ogg/Opus bytes at 16 kHz."""
    if mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if in_sr != OUTPUT_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, in_sr, OUTPUT_SAMPLE_RATE)

    # soundfile expects (frames, channels).
    data = audio.transpose(0, 1).contiguous().cpu().numpy()

    buf = io.BytesIO()
    sf.write(buf, data, OUTPUT_SAMPLE_RATE, format="OGG", subtype="OPUS")
    return buf.getvalue()


def _ogg_to_framed_opus(ogg: bytes) -> bytes:
    """De-encapsulate Ogg/Opus into the linephone wire format.

    Strips the Ogg container, drops the OpusHead/OpusTags header packets, and
    re-frames each remaining raw Opus packet as ``[BE16 length][opus packet]``,
    concatenated. This matches the payload the /api/device/linephone/ endpoint
    consumes -- same Opus packets, no Ogg wrapping.
    """
    packets: list[bytes] = []
    cur = bytearray()
    pos = 0
    n = len(ogg)
    while pos < n:
        if ogg[pos : pos + 4] != b"OggS":
            raise ValueError("not an Ogg bitstream (missing OggS capture pattern)")
        page_segments = ogg[pos + 26]
        seg_table = ogg[pos + 27 : pos + 27 + page_segments]
        seg_off = pos + 27 + page_segments
        for lace in seg_table:
            cur += ogg[seg_off : seg_off + lace]
            seg_off += lace
            # A lacing value < 255 terminates the current packet; 255 means it
            # continues into the next segment (and possibly the next page).
            if lace < 255:
                packets.append(bytes(cur))
                cur = bytearray()
        pos = seg_off

    out = bytearray()
    for pkt in packets:
        # Skip the two Opus header packets; only audio packets go on the wire.
        if pkt.startswith(b"OpusHead") or pkt.startswith(b"OpusTags"):
            continue
        out += struct.pack(">H", len(pkt))
        out += pkt
    return bytes(out)


@app.post("/generate")
async def generate(request: Request):
    params = await _parse_params(request)

    prompt = params.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="`prompt` is required")

    # Optional knobs with sensible defaults (str-safe for form posts).
    duration = float(params.get("duration", 10))
    steps = int(params.get("steps", 8))
    cfg_scale = float(params.get("cfg_scale", 1.0))
    seed = int(params.get("seed", -1))
    negative_prompt = params.get("negative_prompt") or None
    mono = str(params.get("mono", "")).lower() in ("1", "true", "yes", "on")

    def _run() -> bytes:
        with _GEN_LOCK:
            result = MODEL.generate(
                prompt=str(prompt),
                negative_prompt=negative_prompt,
                duration=duration,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
            )
        # result: (batch, channels, samples) -> take first item
        ogg = _encode_ogg_opus(result[0], MODEL.model.sample_rate, mono)
        # Same payload as /api/device/linephone/: framed raw Opus, no Ogg.
        return _ogg_to_framed_opus(ogg)

    # Generation is blocking/CPU+GPU heavy; run it off the event loop.
    opus_bytes = await run_in_threadpool(_run)

    def _stream():
        chunk = 64 * 1024
        for i in range(0, len(opus_bytes), chunk):
            yield opus_bytes[i : i + chunk]

    return StreamingResponse(
        _stream(),
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(len(opus_bytes)),
            # [BE16 length][opus packet]... frames -- not an Ogg/Opus file.
            "X-Audio-Format": "framed-opus",
            "X-Opus-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


def main(args):
    global MODEL
    set_verbose(getattr(args, "verbose", False))
    torch.manual_seed(42)
    MODEL = StableAudioModel.from_pretrained(args.model, model_half=args.model_half)
    if args.lora_ckpt_path:
        MODEL.load_lora(args.lora_ckpt_path)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Stable Audio 3 HTTP API")
    parser.add_argument("--model", type=str, required=True, help="Pretrained model name")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-half", action="store_true", default=True, help="Use half precision"
    )
    parser.add_argument(
        "--lora-ckpt-path", type=str, nargs="*", help="LoRA checkpoint path(s)"
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
