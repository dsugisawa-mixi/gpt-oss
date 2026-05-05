"""
HTTP server for D.Sugisawa's LT presentation system, powered by gpt-oss-20b.

The LLM speaks as D.Sugisawa giving a research presentation. The frontend
controls slide transitions and posts the current slide content to /chat;
the LLM generates speech text per slide which the browser plays via the
existing TTS server alongside a synchronized GLB avatar.

Usage:
    python your_professor_server.py [--checkpoint openai/gpt-oss-20b] [--port 8081]

Endpoints:
    GET  /              — Web UI (waiting screen + GLB viewer + Q&A input)
    POST /chat          — Stage-driven generation (waiting/presenting/qa/closing)
    POST /reset         — Reset conversation history
    GET  /history       — View session history (debug)

Session memory:
    - Short-term: Last 100 turns kept in prompt context
    - Persistence: Full history saved per user_id under professor_data/sessions/
    - RAG context: paper_rag.search() is called per request when stage is
      presenting or qa; absent module = empty knowledge (graceful degrade).
"""

import argparse
import asyncio
import sys
import datetime
import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import boto3
import httpx
import torch  # noqa: F401  (kept so vLLM picks up CUDA env consistently)
from botocore.client import Config as BotoConfig
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
import uvicorn

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
logger = logging.getLogger("professor")

# --- Constants ---
SHORT_TERM_WINDOW = 100
DATA_DIR = Path("professor_data")
SESSIONS_DIR = DATA_DIR / "sessions"

VALID_STAGES = ("waiting", "presenting", "qa", "closing")

# Presence + QA timeline windows.
PRESENCE_TIMEOUT_S = float(os.environ.get("PRESENCE_TIMEOUT_S", "30"))
QA_TIMELINE_MAX = int(os.environ.get("QA_TIMELINE_MAX", "200"))

# Slide generation pipeline (paper PDF -> single-page HTML).
# generate_slides.py reads its OPENAI_API_KEY from a sibling .env and writes
# <stem>_presentation.html next to itself, so we run it with cwd=its dir and
# then copy the output into UPLOAD_DIR for serving.
SLIDE_GEN_SCRIPT = Path(os.environ.get(
    "SLIDE_GEN_SCRIPT",
    "/home/video-dev/git/paper/tools/note/generate_slides.py",
))
SLIDE_HTML2PDF_SCRIPT = Path(os.environ.get(
    "SLIDE_HTML2PDF_SCRIPT",
    "/home/video-dev/git/paper/tools/note/html2pdf.mjs",
))
SLIDE_SCRIPT_GEN_SCRIPT = Path(os.environ.get(
    "SLIDE_SCRIPT_GEN_SCRIPT",
    "/home/video-dev/git/paper/tools/note/generate_scripts.py",
))
SLIDE_GEN_PYTHON = os.environ.get(
    "SLIDE_GEN_PYTHON",
    "/home/video-dev/git/gpt-oss/venv312/bin/python",
)
SLIDE_NODE = os.environ.get("SLIDE_NODE", "node")
UPLOAD_LOG_TAIL = int(os.environ.get("UPLOAD_LOG_TAIL", "200"))
UPLOAD_PDF_MAX_MB = int(os.environ.get("UPLOAD_PDF_MAX_MB", "30"))

# Lab self-description (multi-lab fan-out): tunnel_client fetches
# /api/lab/summary and forwards it to the proxy via update_operator. The
# proxy's /api/tunnel/info exposes it to the GUI for trust-score ranking.
LAB_ID = os.environ.get("LAB_ID", "")
LAB_NAME = os.environ.get("LAB_NAME", "")
LAB_SUMMARY_TEXT = os.environ.get("LAB_SUMMARY", "")
LAB_SUMMARY_MAX_CHARS = int(os.environ.get("LAB_SUMMARY_MAX_CHARS", "200"))

# Browser uploads bypass the WebSocket tunnel (which has a 1 MB frame
# cap) by PUT-ing the PDF directly to S3 with a presigned URL. Only the
# tiny presign/start control-plane requests travel through the tunnel.
S3_UPLOAD_BUCKET = os.environ.get("S3_UPLOAD_BUCKET", "monst-static-assets")
S3_UPLOAD_PREFIX = os.environ.get("S3_UPLOAD_PREFIX", "professor_uploads/")
S3_UPLOAD_REGION = os.environ.get("S3_UPLOAD_REGION", "ap-northeast-1")
S3_PRESIGN_TTL_S = int(os.environ.get("S3_PRESIGN_TTL_S", "300"))

# The proxy in front of S3-hosted frontend only forwards /api/* paths to this
# server through a websocket tunnel; everything else is rejected with 404.
# So all browser-callable endpoints live under /api/.
#
# The browser POSTs to /api/tts/generate_stream which we proxy to TTS_BACKEND.
# /api/config returns this same path so the frontend can stay relative.
TTS_BACKEND = os.environ.get("TTS_BACKEND", "http://192.168.124.251:8889")
TTS_BACKEND_PATH = os.environ.get("TTS_BACKEND_PATH", "/api/generate_stream")
TTS_TIMEOUT_S = float(os.environ.get("TTS_TIMEOUT_S", "10"))
TTS_PUBLIC_PATH = "/api/tts/generate_stream"

# Google OAuth — same client id as the existing nanny game uses (myboy/js/auth.js).
GOOGLE_CLIENT_ID = os.environ.get(
    "GOOGLE_CLIENT_ID",
    "349549531314-2s0rvdf4hsb92l2hae14viintjjpvl5t.apps.googleusercontent.com",
)
# Allow only mixi.co.jp and its subdomains. Set ALLOWED_EMAIL_DOMAIN to override.
ALLOWED_EMAIL_DOMAIN = os.environ.get("ALLOWED_EMAIL_DOMAIN", "mixi.co.jp")
# Optional dev allowlist + role grants. Each non-comment, non-blank line:
#     <email> [role1,role2,...]
# - Email-only lines bypass the ALLOWED_EMAIL_DOMAIN check.
# - The optional second token is a comma-separated list of roles
#   (no spaces inside) granted to that email. Roles gate role-protected
#   endpoints — currently "ticket-admin" for /api/ticket/.
# - Domain-allowed users may also appear here purely to receive roles;
#   their listing has no effect on the domain check for anyone else.
CUSTOM_AUTH_FILE = Path(os.environ.get("CUSTOM_AUTH_FILE", ".custom-auth.txt"))
custom_allowlist: set[str] = set()
custom_roles: dict[str, set[str]] = {}

SYSTEM_PROMPT_TEMPLATE = """\
# 役割
あなたは {presenter_line}本人として、研究発表(LT)をおこなう。
聴衆に対して一人称で語り、自分の研究を自分の言葉で説明する。
出力テキストは TTS で読み上げられ、ブラウザ上の GLB アバターがスライド遷移と
同期してそれを話している演出となる。
{theme_section}
# 発話モード
リクエストには [Stage] セクションが付与され、現在の発表段階が示される。
モードに応じて応答を切り替える。

- waiting(開始前)
  発表タイトルを丁寧に案内し、「もう少ししましたら開始いたします、もうしばらく
  お待ちください」のように締める。具体的な時刻(13:00 等)は書かない。1〜2 文。
- presenting(発表中)
  [Slide] セクションの現在ページに記載された主張・要点を、順序立てて自分の言葉で
  説明する。スライドに書かれていないことを勝手に追加しない。3〜6 文を目安に、
  そのページの主張が伝わる長さで。
- qa(質疑応答)
  まず聴衆の質問の意図を捉え、それに対して直接応答する。
  質問が「○○ですよね?」のような確認なら、是/否をはっきり示す。
  「フツー」「当たり前」「なぜ○○と比較したのか」など反論的・批判的な質問には、
  まずその指摘を受け止め、自分の研究の独自性(なぜこの比較・この設計なのか)を
  1〜2文で述べる。資料外・未検証の事項は「その点は本日の発表範囲外」
  「現在検証中」と率直に言う。発表内容の要約をだらだら繰り返してはならない。
  3〜5 文。
- closing(締め)
  「ご清聴ありがとうございました」で短く締めくくる。1 文。

# 出力ルール(TTS 前提)
- 自然な日本語の話し言葉で書く。一人称は「私」、聴衆への呼びかけは控えめに。
- マークダウン記号(#, *, -)、箇条書き、絵文字、コードブロックは使わない。
  読み上げで意味をなさないため。
- 数式は言葉で説明する。
- 定着した略語(TCP, RTT, WebRTC など)はそのままで構わないが、馴染みのない
  記号や英単語は読みやすく言い換える。
- セクション見出しや「以下の通りです」といった文書的な前置きは使わない。

# 知識ソース
- [Knowledge Context] セクションには、自分の論文・メモ・発表資料(公開・未公開を含む)
  から検索された関連抜粋が入る。これは自分自身の過去の記述として参照してよい。
- 抜粋に書かれていない実験結果・数値・主張を捏造してはいけない。該当情報が
  なければ範囲外として切り上げる。
- 抜粋とスライド本文が矛盾するときは、スライド本文を優先する。

# 禁止事項
- 発表テーマと無関係な雑談はしない。
- 他者の研究を批判的に語らない。比較は中立的に述べる。
- スライド外の数値・図表・引用を勝手に作らない。
- 自分が話していない内容について「先ほど述べたように」と虚構の参照をしない。"""


_DEFAULT_PRESENTER = "D.Sugisawa(杉澤 大輔, mixi.co.jp 所属)"


def build_system_prompt(theme: str, presenter: str = "",
                        affiliation: str = "", venue: str = "") -> str:
    # -- presenter line ------------------------------------------------
    if presenter.strip():
        presenter_line = presenter.strip()
        if affiliation.strip():
            presenter_line += f"({affiliation.strip()} 所属)"
    else:
        presenter_line = _DEFAULT_PRESENTER
    if venue.strip():
        presenter_line += f"({venue.strip()})"

    # -- theme section -------------------------------------------------
    if theme:
        theme_section = f"\n# 発表テーマ\n「{theme}」\n"
    else:
        theme_section = ""

    return SYSTEM_PROMPT_TEMPLATE.format(
        presenter_line=presenter_line,
        theme_section=theme_section,
    )


# =====================================================================
# FastAPI app
# =====================================================================

app = FastAPI(title="Professor LT Server")

# CORS: the frontend ships from S3/CloudFront, the API runs here. Allow the
# specific origins via env (comma-separated) or fall back to "*" for dev.
_cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    raw_errors = exc.errors()
    safe_errors = []
    for e in raw_errors:
        e = dict(e)
        if isinstance(e.get("input"), (bytes, bytearray)):
            try:
                e["input"] = bytes(e["input"]).decode("utf-8", errors="replace")
            except Exception:
                e["input"] = repr(e["input"])
        safe_errors.append(e)
    logger.error("Validation error on %s %s: %s", request.method, request.url.path, safe_errors)
    logger.error("Request body: %s", exc.body)
    return JSONResponse(status_code=422, content={"detail": safe_errors})


@app.middleware("http")
async def log_response_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s %d %.1fms",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.middleware("http")
async def gate_until_rag_ready(request: Request, call_next):
    """Block every /api/* request with 503 until the startup RAG rebuild
    has finished. Lets non-/api paths and CORS preflight through so the
    static UI can still load and browsers can negotiate CORS while we
    warm up. The proxy registry treats /api/lab/summary 503 as
    'unavailable' and omits this lab from the LAB list."""
    if (
        not _rag_ready
        and request.method != "OPTIONS"
        and request.url.path.startswith("/api/")
    ):
        logger.info("gate: 503 (rag not ready) %s %s",
                    request.method, request.url.path)
        return JSONResponse(
            status_code=503,
            content={"detail": "lab warming up: rag index rebuild in progress"},
            headers={"Retry-After": "10"},
        )
    return await call_next(request)


# Global state
generator = None   # vLLM TokenGenerator
encoding = None    # Harmony encoding
sessions: dict[str, list[dict[str, str]]] = {}

# In-memory room state (NOT persisted; cleared on every server restart).
# This single dict drives both the participants sidebar and the QA timeline:
#   participants[user_id] = {
#       "user_id", "display_name",
#       "joined_at", "last_seen",
#       "qa": [qa_entry, ...]      # this user's contributions, append-only
#   }
# A user whose last_seen is older than PRESENCE_TIMEOUT_S is considered
# offline but their record (and qa list) stays in the dict so timeline
# entries don't disappear when someone briefly disconnects.
participants: dict[str, dict] = {}

# Globally unique, monotonically increasing QA id, scoped to the current
# server process. Reset on restart, by design.
_qa_id_counter = 0

# Active presentation metadata — populated from the most recently uploaded
# paper. `theme` is injected into the system prompt; `presenter` and
# `venue` are echoed to the browser header. All cleared on restart.
current_meta: dict[str, str] = {"theme": "", "presenter": "", "affiliation": "", "venue": ""}

# Paper-upload jobs. Cleared on restart, by design.
#   upload_jobs[job_id] = {
#       "job_id", "user_id", "filename", "status",
#       "log": [str, ...],          # tail of subprocess stdout/stderr
#       "started_at", "finished_at",
#       "pdf_path": Path,           # uploaded copy under DATA_DIR
#       "html_path": Optional[Path],# generated slide HTML
#       "essence_path": Optional[Path],
#       "error": Optional[str],
#   }
upload_jobs: dict[str, dict] = {}


# =====================================================================
# Pydantic models
# =====================================================================

class SlideContent(BaseModel):
    page: int
    title: Optional[str] = None
    bullets: Optional[list[str]] = None
    notes: Optional[str] = None


class ChatRequest(BaseModel):
    user_id: str
    message: str = ""
    stage: str = "waiting"            # waiting | presenting | qa | closing
    slide: Optional[SlideContent] = None
    voice: Optional[str] = None       # TTS voice override
    ephemeral: bool = False           # don't persist this turn


class ChatResponse(BaseModel):
    user_id: str
    reply: str
    stage: str
    slide_page: Optional[int] = None
    voice: str = "male"
    # Lab self-id (LAB_ID env). The GUI fans out qa-stage chat to multiple
    # Labs in parallel; this lets each response carry its origin without the
    # client having to remember "which operator did I send this to".
    lab_id: str = ""
    # Per-response trust signal. mean/top scores come from the RAG hits
    # used as grounding context; rag_hits is the count of hits that fit
    # within top_k. Empty knowledge ⇒ all zeros (presenter narration with
    # only slide context, or an empty RAG index).
    accuracy: dict = {}


class GoogleAuthRequest(BaseModel):
    email: str
    sub: str
    credential: str
    displayName: Optional[str] = None
    lang: Optional[str] = None


class GoogleAuthResponse(BaseModel):
    userId: str
    displayName: str
    lang: Optional[str] = None


class HeartbeatRequest(BaseModel):
    user_id: str


class QAEditRequest(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None


class UploadPresignRequest(BaseModel):
    filename: str


class UploadStartRequest(BaseModel):
    job_id: str


# =====================================================================
# Session persistence  (full history saved to disk, keyed by user_id)
# =====================================================================

def _session_path(user_id: str) -> Path:
    return SESSIONS_DIR / f"{user_id}.json"


def load_session(user_id: str) -> list[dict[str, str]]:
    path = _session_path(user_id)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_session(user_id: str, history: list[dict[str, str]]):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_session_path(user_id), "w") as f:
        json.dump(history, f, ensure_ascii=False)


def get_or_create_session(user_id: str) -> list[dict[str, str]]:
    if user_id in sessions:
        return sessions[user_id]
    history = load_session(user_id)
    sessions[user_id] = history
    return history


# =====================================================================
# Auth  (Google Sign-In; allow only @<ALLOWED_EMAIL_DOMAIN> and subdomains)
# =====================================================================

def _users_path() -> Path:
    return DATA_DIR / "users.json"


def _load_users() -> dict:
    p = _users_path()
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        logger.exception("failed to read users.json")
        return {}


def _save_users(users: dict) -> None:
    p = _users_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _load_custom_auth() -> tuple[set[str], dict[str, set[str]]]:
    """Read CUSTOM_AUTH_FILE → (allowlist, roles).

    Each non-comment, non-blank line is `<email> [role1,role2,...]`. The
    email goes into the allowlist (lowercased) regardless of whether
    roles are present; the second token (if any) is split on ',' into a
    set of role names mapped to that email. Missing file → ({}, {})."""
    emails: set[str] = set()
    roles: dict[str, set[str]] = {}
    if not CUSTOM_AUTH_FILE.exists():
        return emails, roles
    try:
        for line in CUSTOM_AUTH_FILE.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(None, 1)
            email = parts[0].lower()
            emails.add(email)
            if len(parts) == 2:
                role_set = {
                    r.strip() for r in parts[1].split(",") if r.strip()
                }
                if role_set:
                    roles[email] = role_set
    except Exception:
        logger.exception("failed to read %s", CUSTOM_AUTH_FILE)
    return emails, roles


def _is_allowed_email(email: str) -> bool:
    """True iff email's domain is ALLOWED_EMAIL_DOMAIN (or subdomain of it),
    OR the email is explicitly listed in CUSTOM_AUTH_FILE."""
    if not email or "@" not in email:
        return False
    e = email.lower()
    if e in custom_allowlist:
        return True
    domain = e.rsplit("@", 1)[1]
    allowed = ALLOWED_EMAIL_DOMAIN.lower()
    return domain == allowed or domain.endswith("." + allowed)


# ---- Presence + QA timeline (single unified `participants` dict) ----

def _is_online(rec: dict) -> bool:
    return (time.time() - rec.get("last_seen", 0)) <= PRESENCE_TIMEOUT_S


def _touch_presence(user_id: str) -> Optional[dict]:
    """Register a user (or refresh their last_seen). Looks up display name
    from the persistent users.json (filled in by /api/auth/google)."""
    users = _load_users()
    user = users.get(user_id)
    if not user:
        return None
    now = time.time()
    rec = participants.get(user_id)
    if rec is None:
        rec = {
            "user_id": user_id,
            "display_name": user["display_name"],
            "joined_at": now,
            "last_seen": now,
            "qa": [],
        }
        participants[user_id] = rec
        logger.info("[presence] join: %s (%s)", user_id, user["display_name"])
    else:
        rec["last_seen"] = now
        rec["display_name"] = user["display_name"]
    return rec


def _record_qa(
    *,
    user_id: str,
    question: str,
    answer: str,
    slide_page: Optional[int],
    lab_id: str = "",
    accuracy: Optional[dict] = None,
) -> dict:
    """Append one Q&A exchange under the asking user's record. The qa_id is
    process-globally unique within this Lab; it is NOT unique across Labs,
    so multi-Lab clients merge timelines by `ts`. lab_id + accuracy are
    persisted so the merged GUI can label items and rank parallel answers."""
    global _qa_id_counter
    _qa_id_counter += 1
    rec = _touch_presence(user_id) or participants.get(user_id)
    display_name = (rec or {}).get("display_name", user_id[:8])
    entry = {
        "qa_id": _qa_id_counter,
        "user_id": user_id,
        "display_name": display_name,
        "question": question,
        "answer": answer,
        "slide_page": slide_page,
        "ts": time.time(),
        "lab_id": lab_id,
        "accuracy": accuracy or {},
    }
    if rec is not None:
        rec.setdefault("qa", []).append(entry)
    return entry


def _all_qa_entries() -> list[dict]:
    """Flatten all participants' qa lists into one list."""
    out: list[dict] = []
    for rec in participants.values():
        out.extend(rec.get("qa", []))
    return out


def _find_qa(qa_id: int):
    """Return (participant_rec, index_in_qa_list) or (None, -1)."""
    for rec in participants.values():
        qa = rec.get("qa", [])
        for i, e in enumerate(qa):
            if e["qa_id"] == qa_id:
                return rec, i
    return None, -1


def require_auth(request: Request) -> str:
    """FastAPI dependency: require a valid X-User-Id header that resolves to
    a registered user (i.e. one who passed Google sign-in + domain check)."""
    uid = request.headers.get("X-User-Id") or request.headers.get("x-user-id")
    if not uid:
        raise HTTPException(status_code=401, detail="missing X-User-Id header")
    if uid not in _load_users():
        raise HTTPException(status_code=401, detail="unauthenticated")
    return uid


def _user_roles(user_id: str) -> set[str]:
    """Roles granted to a registered user via CUSTOM_AUTH_FILE. Looked up
    by email (case-insensitive). Empty set if the user has no entry or
    no roles."""
    email = _user_email(user_id)
    if not email:
        return set()
    return set(custom_roles.get(email, set()))


def require_role(role: str):
    """Build a FastAPI dependency that requires the caller to be
    authenticated AND to have the given role assigned in
    CUSTOM_AUTH_FILE. Returns the user_id on success."""
    def dep(uid: str = Depends(require_auth)) -> str:
        if role not in _user_roles(uid):
            raise HTTPException(
                status_code=403,
                detail=f"role required: {role}",
            )
        return uid
    return dep


# =====================================================================
# Tickets  (paid feature gate, single-use, per-action)
# =====================================================================
# Two distinct paid actions — paper upload and paper deletion — each
# require their own single-use ticket. Tickets live on disk (currently
# placed by hand; eventually by the App Store receipt verifier) as
# one JSON file per purchase:
#
#     TICKETS_DIR/<uuid>.ticket.available   <- spends on /api/upload/start
#     TICKETS_DIR/<uuid>.ticket.remove      <- spends on DELETE /api/upload/papers/{id}
#     contents: {"email": "<google account>", "action": "upload"|"remove", ...}
#
# When the gated action runs, the matching ticket file is atomically
# renamed to <uuid>.ticket.consumed (audit trail — keeps the original
# JSON inside, which still says action="upload" or "remove" so post-hoc
# inspection knows what was spent on what).
#
# We re-read the directory on every check (no caching) so a newly
# placed ticket takes effect immediately and a freshly consumed one
# stops counting, without needing a server restart.
TICKETS_DIR = Path(os.environ.get("TICKETS_DIR", str(DATA_DIR / "tickets")))
TICKET_CONSUMED_SUFFIX = ".ticket.consumed"
# Map action name -> filename suffix. Adding a new paid action means
# adding one entry here and one require_<action>_ticket dependency.
TICKET_ACTION_SUFFIX = {
    "upload": ".ticket.available",
    "remove": ".ticket.remove",
}


def _ticket_email(path: Path) -> str:
    """Read a ticket JSON and return its lowercase email, or "" on
    parse failure. Accepts either `email` or `google_account` as the
    field name to match either the App Store receipt schema or the
    in-house format."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return (data.get("email") or data.get("google_account") or "").strip().lower()


def _list_available_tickets(action: str) -> list[Path]:
    """All currently-available ticket files for the given action,
    oldest first (FIFO consumption order). Returns [] for unknown
    actions or when TICKETS_DIR doesn't exist."""
    suffix = TICKET_ACTION_SUFFIX.get(action)
    if suffix is None or not TICKETS_DIR.exists():
        return []
    paths = list(TICKETS_DIR.glob(f"*{suffix}"))
    paths.sort(key=lambda p: p.stat().st_mtime)
    return paths


def _load_ticket_emails(action: str) -> set[str]:
    """Set of lowercase emails that have at least one ticket of the
    given action. Used by the eligibility endpoint and the
    require_*_ticket dependencies."""
    out: set[str] = set()
    for p in _list_available_tickets(action):
        e = _ticket_email(p)
        if e:
            out.add(e)
        else:
            logger.warning("ticket: %s has no email/google_account; ignoring", p.name)
    return out


def _consume_ticket_for(email: str, action: str) -> Optional[Path]:
    """Atomically claim one ticket of the given action matching `email`
    by renaming to <uuid>.ticket.consumed. Returns the renamed path on
    success, None if no matching ticket existed (or all attempts lost
    the rename race to a concurrent consumer)."""
    target = email.strip().lower()
    if not target:
        return None
    suffix = TICKET_ACTION_SUFFIX.get(action)
    if suffix is None:
        return None
    for p in _list_available_tickets(action):
        if _ticket_email(p) != target:
            continue
        stem = p.name.removesuffix(suffix)
        consumed = p.with_name(stem + TICKET_CONSUMED_SUFFIX)
        try:
            # Path.rename is atomic on POSIX — if two requests race for
            # the same ticket, only one rename succeeds; the other
            # raises FileNotFoundError and we move on to the next.
            p.rename(consumed)
            return consumed
        except FileNotFoundError:
            continue
        except Exception:
            logger.exception("ticket: rename failed for %s", p)
            return None
    return None


def _user_email(user_id: str) -> str:
    """Look up the email for a registered user_id (UUID). Returns ""
    if the user doesn't exist or has no email recorded."""
    users = _load_users()
    return ((users.get(user_id) or {}).get("email") or "").strip().lower()


# ---- Ticket admin CRUD helpers ----

def _action_for_suffix(suffix: str) -> Optional[str]:
    """Reverse-map a filename suffix (e.g. '.ticket.available') to its
    action name. Returns None for '.ticket.consumed' — consumed tickets
    keep the action only inside the JSON body."""
    for action, sfx in TICKET_ACTION_SUFFIX.items():
        if sfx == suffix:
            return action
    return None


def _ticket_path_for(ticket_id: str) -> Optional[Path]:
    """Locate the on-disk ticket file by its uuid, regardless of status.
    Returns the first match (uuids are globally unique). None if no
    match or TICKETS_DIR is missing. Rejects ids containing path
    separators so callers can pass user input directly."""
    if not TICKETS_DIR.exists() or "/" in ticket_id or ".." in ticket_id:
        return None
    matches = list(TICKETS_DIR.glob(f"{ticket_id}.ticket.*"))
    return matches[0] if matches else None


def _ticket_view(path: Path) -> dict:
    """Render one ticket file as the /api/ticket/ API view. The action
    is derived from the filename suffix when available (most accurate
    for non-consumed tickets) and falls back to the JSON body for
    consumed tickets where the suffix is just '.ticket.consumed'."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    name = path.name
    if ".ticket." in name:
        ticket_id, tail = name.split(".ticket.", 1)
        suffix = ".ticket." + tail
    else:
        ticket_id, suffix = name, ""
    if suffix == TICKET_CONSUMED_SUFFIX:
        status = "consumed"
        action = data.get("action") or ""
    else:
        status = "available"
        action = _action_for_suffix(suffix) or data.get("action") or ""
    return {
        "ticket_id": ticket_id,
        "action": action,
        "status": status,
        "email": (data.get("email") or data.get("google_account") or "").strip().lower(),
        "note": data.get("note") or "",
        "purchased_at": data.get("purchased_at") or "",
        "transaction_id": data.get("transaction_id") or "",
        "created_at": data.get("created_at") or 0,
    }


def _list_all_ticket_paths() -> list[Path]:
    """Every ticket file in TICKETS_DIR — available + consumed."""
    if not TICKETS_DIR.exists():
        return []
    return list(TICKETS_DIR.glob("*.ticket.*"))


def _user_has_ticket(user_id: str, action: str) -> bool:
    """True iff the user is signed in AND has at least one ticket of
    the given action. Empty email or missing ticket -> False."""
    email = _user_email(user_id)
    if not email:
        return False
    return email in _load_ticket_emails(action)


def _require_ticket_for(action: str, error_msg: str):
    """Build a FastAPI dependency that requires a ticket of the given
    action. Soft pre-check — the actual consumption happens inside the
    handler via _consume_ticket_for(), so a concurrent caller may still
    take the last ticket between this check and the consume."""
    def dep(uid: str = Depends(require_auth)) -> str:
        if not _user_has_ticket(uid, action):
            raise HTTPException(status_code=403, detail=error_msg)
        return uid
    return dep


require_ticket = _require_ticket_for(
    "upload",
    "no ticket: 論文アップロード用チケット (.ticket.available) がありません",
)
require_remove_ticket = _require_ticket_for(
    "remove",
    "no ticket: 論文削除用チケット (.ticket.remove) がありません",
)


def _verify_google_credential(credential: str) -> dict:
    """Verify a Google ID token and return its decoded payload.

    Uses google-auth when available (full signature + audience check). Falls
    back to unsigned decode with a warning if google-auth is missing.
    """
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
        return id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ImportError:
        logger.warning("google-auth missing; decoding JWT without signature check")
        import base64
        parts = credential.split(".")
        if len(parts) < 2:
            raise ValueError("malformed JWT")
        payload = parts[1]
        payload += "=" * ((4 - len(payload) % 4) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))


# =====================================================================
# RAG hook  (paper_rag.search() is wired here once available)
# =====================================================================

try:
    import paper_rag as _paper_rag  # type: ignore
    _rag_search = _paper_rag.search
except Exception:
    _paper_rag = None
    _rag_search = None

# FIFO queue of rebuild triggers. A single long-running worker task
# (_rag_rebuild_worker, spawned at startup) drains the queue and runs one
# rebuild + swap pass per item. This guarantees per-upload rebuilds run
# in arrival order with no concurrency between them — so search() never
# observes a torn state and no upload's rebuild gets dropped.
_rag_rebuild_queue: "asyncio.Queue[str]" = None  # initialized at startup

# Service-readiness gate. False until the worker has drained the
# "startup" trigger enqueued by _enqueue_startup_rebuild — i.e. until
# build_paper_index.py has finished and the in-memory index has been
# swapped in. While False the gate_until_rag_ready middleware returns
# 503 for every /api/* request (incl. /api/lab/summary), so the
# multi-lab registry omits this lab from the LAB list. If paper_rag
# failed to import we leave this False forever, by design — the lab
# should drop out of the registry rather than serve degraded.
_rag_ready: bool = False


@app.on_event("startup")
async def _preload_rag():
    """Warm the embedder + index so the first /chat doesn't stall ~9s.

    If preload finds a usable on-disk index, flip _rag_ready immediately
    so /api/* unblocks without waiting for the startup rebuild — that
    rebuild can take minutes on CPU and there's no point gating
    availability behind it when we already have a functional index.
    The startup rebuild still runs in the background to pick up any
    new uploads since the last build, and reload_index() atomically
    swaps the result in when it finishes."""
    global _rag_ready
    if _paper_rag is None:
        return
    try:
        ok = _paper_rag.preload()
        logger.info("paper_rag preload: %s", "ready" if ok else "unavailable")
        if ok and not _rag_ready:
            _rag_ready = True
            logger.info(
                "preload ready → /api/* unblocked (startup rebuild "
                "continues in background)",
            )
    except Exception:
        logger.exception("paper_rag preload failed")


@app.on_event("startup")
async def _start_rag_rebuild_worker():
    """Initialize the rebuild queue (must be done in the running event
    loop) and spawn the single FIFO worker that drains it."""
    global _rag_rebuild_queue
    if _paper_rag is None:
        return
    _rag_rebuild_queue = asyncio.Queue()
    asyncio.create_task(_rag_rebuild_worker())


@app.on_event("startup")
async def _enqueue_startup_rebuild():
    """Force a fresh build_paper_index.py run on every server start, and
    keep /api/* gated behind 503 until it finishes. Ordering: this runs
    after _start_rag_rebuild_worker, so the queue and worker exist by
    the time we put_nowait()."""
    if _paper_rag is None:
        # Stay un-ready forever — the gate middleware will 503 every
        # /api/* call so the proxy registry drops this lab.
        return
    _enqueue_rag_rebuild("startup")


async def _do_rag_rebuild_once(trigger: str) -> bool:
    """One pass of subprocess-build + in-memory reload. Returns True iff
    both build_paper_index.py succeeded AND reload_index swapped cleanly.
    Caller must hold _rag_rebuild_lock."""
    if _paper_rag is None:
        return False
    # build_paper_index.py defaults to ~/git/paper, which is the host
    # layout. In container the paper tree is bind-mounted elsewhere
    # (default /app/paper, overridable via PAPER_DIR env). Pass it
    # explicitly so the rebuild works in both setups.
    paper_dir = os.environ.get("PAPER_DIR", "/app/paper")
    cmd = [sys.executable, "build_paper_index.py", "--paper-dir", paper_dir]
    logger.info(
        "rag rebuild (%s): starting build_paper_index.py --paper-dir %s ...",
        trigger, paper_dir,
    )
    t0 = time.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out_bytes, _ = await proc.communicate()
    except Exception:
        logger.exception("rag rebuild (%s): subprocess launch failed", trigger)
        return False
    if proc.returncode != 0:
        tail = out_bytes.decode("utf-8", errors="replace")[-1000:]
        logger.warning(
            "rag rebuild (%s): build_paper_index.py exit=%d, tail=%s",
            trigger, proc.returncode, tail,
        )
        return False
    logger.info(
        "rag rebuild (%s): build done in %.1fs; reloading into memory ...",
        trigger, time.time() - t0,
    )
    # reload_index() is blocking (loads SentenceTransformer + opens
    # LanceDB), so push it to the default thread executor so the
    # event loop stays responsive for /chat etc.
    try:
        ok = await asyncio.get_running_loop().run_in_executor(
            None, _paper_rag.reload_index,
        )
    except Exception:
        logger.exception("rag rebuild (%s): reload_index raised", trigger)
        return False
    logger.info(
        "rag rebuild (%s): swap %s, total %.1fs",
        trigger, "ok" if ok else "failed", time.time() - t0,
    )
    return bool(ok)


def _enqueue_rag_rebuild(trigger: str) -> None:
    """Append a rebuild request to the FIFO queue. Non-blocking; the worker
    drains them in arrival order. Safe to call before the worker has
    started (the queue is initialized at module load) — items just sit
    until startup spawns the worker."""
    if _paper_rag is None or _rag_rebuild_queue is None:
        logger.info("rag rebuild (%s): paper_rag unavailable; skipping", trigger)
        return
    _rag_rebuild_queue.put_nowait(trigger)
    logger.info(
        "rag rebuild (%s): queued (depth=%d)",
        trigger, _rag_rebuild_queue.qsize(),
    )


async def _rag_rebuild_worker() -> None:
    """Single consumer of the rebuild queue. Runs forever; one rebuild +
    swap per dequeued trigger, strict FIFO. The first time the
    "startup" trigger is processed, flip _rag_ready True so the
    gate middleware lets /api/* through — regardless of build success,
    since failure leaves the previous on-disk index in place and we
    prefer "serve with stale index" over "block forever"."""
    global _rag_ready
    logger.info("rag rebuild worker: started")
    while True:
        trigger = await _rag_rebuild_queue.get()
        try:
            ok = await _do_rag_rebuild_once(trigger)
        except Exception:
            logger.exception("rag rebuild worker: pass for %s raised", trigger)
            ok = False
        finally:
            _rag_rebuild_queue.task_done()
        if trigger == "startup" and not _rag_ready:
            _rag_ready = True
            logger.info(
                "rag rebuild worker: startup pass done (ok=%s); /api/* unblocked",
                ok,
            )


@app.on_event("startup")
async def _load_auth_allowlist():
    """Load the dev email allowlist + role grants (CUSTOM_AUTH_FILE)."""
    global custom_allowlist, custom_roles
    custom_allowlist, custom_roles = _load_custom_auth()
    if custom_allowlist:
        logger.info(
            "custom auth: %d email(s), %d with roles, from %s",
            len(custom_allowlist), len(custom_roles), CUSTOM_AUTH_FILE,
        )
    else:
        logger.info("custom auth: none (%s missing or empty)", CUSTOM_AUTH_FILE)


@app.on_event("startup")
async def _rehydrate_uploaded_papers():
    """Pick up previously-uploaded papers from disk so they show up in
    /api/upload/papers without re-running the generation pipeline."""
    n = _rehydrate_papers_from_disk()
    if n:
        logger.info("rehydrated %d uploaded paper(s) from %s", n, _upload_dir())


def retrieve_knowledge(query: str, *, top_k: int = 5) -> list[dict]:
    """Return [{title, source, text}, ...] or [] if RAG unavailable."""
    if not _rag_search or not query.strip():
        return []
    try:
        return _rag_search(query, top_k=top_k)
    except Exception:
        logger.exception("paper_rag.search failed; returning empty knowledge")
        return []


def _build_rag_query(stage: str, message: str, slide: Optional[SlideContent]) -> str:
    parts: list[str] = []
    if slide:
        if slide.title:
            parts.append(slide.title)
        if slide.bullets:
            parts.extend(slide.bullets)
    if stage == "qa" and message:
        parts.append(message)
    return " ".join(p for p in parts if p).strip()


# =====================================================================
# Prompt construction  (system + 3 sections + short-term window)
# =====================================================================

def build_prompt_messages(
    user_id: str,
    history: list[dict[str, str]],
    *,
    stage: str = "waiting",
    slide: Optional[dict] = None,
    knowledge: Optional[list[dict]] = None,
) -> list[dict[str, str]]:
    """
    Build the message list sent to the model:
      1. system prompt  (LT presenter persona, with the active theme
         injected when one has been set via /api/meta)
      2. [Stage]            — always appended (waiting/presenting/qa/closing)
      3. [Slide]            — appended when slide info is provided
      4. [Knowledge Context] — appended when RAG returned hits
      5. Last SHORT_TERM_WINDOW history messages (with merge of consecutive
         same-role turns and a skip-notice if history is longer)
    """
    system_content = build_system_prompt(
        theme=current_meta.get("theme", ""),
        presenter=current_meta.get("presenter", ""),
        affiliation=current_meta.get("affiliation", ""),
        venue=current_meta.get("venue", ""),
    )

    system_content += f"\n\n[Stage] {stage}\n"

    if slide:
        page = slide.get("page", "?")
        title = (slide.get("title") or "").strip()
        bullets = slide.get("bullets") or []
        notes = (slide.get("notes") or "").strip()
        slide_lines = [f"\n[Slide] page={page}"]
        if title:
            slide_lines.append(f"title: {title}")
        if bullets:
            slide_lines.append("points:")
            slide_lines.extend(f"- {b}" for b in bullets)
        if notes:
            slide_lines.append("notes:")
            slide_lines.append(notes)
        system_content += "\n".join(slide_lines) + "\n"

    if knowledge:
        kc_lines = [
            "\n[Knowledge Context]",
            "以下は私自身の論文・メモ・発表資料からの抜粋です。発言の根拠として参照してよい。",
            "ただし抜粋に書かれていない数値や主張を捏造してはならない。",
        ]
        for i, doc in enumerate(knowledge, 1):
            title = doc.get("title", "untitled")
            source = doc.get("source", "")
            text = (doc.get("text") or "").strip()
            head = f"\n## Source {i}: {title}"
            if source:
                head += f" ({source})"
            kc_lines.append(head)
            if text:
                kc_lines.append(text)
        system_content += "\n".join(kc_lines) + "\n"

    messages = [{"role": "system", "content": system_content}]

    if len(history) > SHORT_TERM_WINDOW:
        skipped = len(history) - SHORT_TERM_WINDOW
        messages.append({
            "role": "system",
            "content": f"[{skipped} earlier messages omitted.]",
        })
        recent = history[-SHORT_TERM_WINDOW:]
    else:
        recent = history

    sanitized: list[dict[str, str]] = []
    for msg in recent:
        content = msg.get("content", "").strip()
        if not content:
            continue
        if sanitized and sanitized[-1]["role"] == msg["role"]:
            sanitized[-1] = {
                "role": msg["role"],
                "content": sanitized[-1]["content"] + "\n" + content,
            }
        else:
            sanitized.append({"role": msg["role"], "content": content})

    messages.extend(sanitized)
    logger.info(
        "prompt: user=%s stage=%s slide=%s know=%d hist=%d sending=%d",
        user_id, stage,
        (slide or {}).get("page", "-"),
        len(knowledge or []),
        len(history),
        len(messages),
    )
    return messages


# =====================================================================
# Generation
# =====================================================================

def _to_harmony_messages(prompt_messages: list[dict]) -> list[Message]:
    """Convert simple role/content dicts to Harmony Message objects.

    Supported dict shapes:
      {"role": "system",    "content": str}
      {"role": "user",      "content": str}
      {"role": "assistant", "content": str}
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
                    .with_reasoning_effort(ReasoningEffort.MEDIUM)
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
                Message.from_role_and_content(Role.ASSISTANT, content).with_channel("final")
            )

    return harmony_msgs


def generate_reply(
    prompt_messages: list[dict],
    max_new_tokens: int = 512,
    max_reply_tokens: int = 220,
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

    decoded_prompt = encoding.decode(tokens)
    logger.info("=== PROMPT DUMP (last 500 chars) ===\n%s", decoded_prompt[-500:])

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

    for predicted_token in generator.generate(
        tokens, stop_tokens=stop_tokens, temperature=0.7, max_tokens=max_new_tokens
    ):
        parser.process(predicted_token)
        gen_count += 1
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
    logger.info(
        "generate: reply_len=%d truncated=%s in=%d out=%d (final=%d) "
        "encode=%.1fms gen=%.1fms (%.1f tok/s) channels=%s",
        len(reply), truncated, len(tokens), gen_count, final_token_count,
        (t1 - t0) * 1000, gen_ms, tok_per_sec, channels_seen,
    )
    return reply, truncated


# =====================================================================
# Endpoints
# =====================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    logger.info("chat: %s", req.model_dump())

    # Auth: only registered users (= signed in via Google with @mixi.co.jp
    # domain) may invoke chat. The cost of generation + TTS justifies the gate.
    users = _load_users()
    if req.user_id not in users:
        logger.info("chat: rejecting unauthenticated user_id=%s", req.user_id)
        return JSONResponse(
            status_code=401,
            content={"detail": "unauthenticated; sign in via /api/auth/google first"},
        )

    if req.stage not in VALID_STAGES:
        return JSONResponse(
            status_code=400,
            content={"detail": f"invalid stage: {req.stage!r}. expected one of {VALID_STAGES}"},
        )

    # Only `presenting` is stateful — earlier slides should inform later ones
    # (so the model doesn't repeat ideas across the deck). waiting / qa /
    # closing are independent: each invocation is a self-contained generation
    # over [Stage] + [Slide?] + [Knowledge?] + the synthesized user cue, with
    # no carryover. This avoids the echo loop where a prior long answer was
    # being parroted back as a "waiting" greeting.
    stateful = req.stage == "presenting"

    history = get_or_create_session(req.user_id)
    if req.ephemeral or not stateful:
        history = list(history)  # shallow copy — session untouched

    # Build the user-facing turn that anchors this generation. Some stages
    # (e.g. presenting on slide-change) carry no real message; synthesize a
    # short cue so the conversation has an alternating user/assistant shape.
    user_text = (req.message or "").strip()
    if not user_text:
        if req.stage == "presenting" and req.slide:
            user_text = f"[次のスライドへ進みました。page={req.slide.page}]"
        elif req.stage == "waiting":
            user_text = "[聴衆として開始を待っています]"
        elif req.stage == "closing":
            user_text = "[締めの挨拶をお願いします]"
        else:
            user_text = "[継続]"
    elif req.stage == "qa":
        # QA: small LLMs tend to summarise the surrounding [Knowledge Context]
        # instead of answering a short, ambiguous question. Wrap the question
        # so the directive ("respond directly to the question") sits in the
        # user turn itself, where the model is least likely to ignore it.
        user_text = (
            "【聴衆からの質問】\n"
            f"{user_text}\n\n"
            "上の質問の意図をまず捉え、直接応答すること。"
            "発表内容の要約や Knowledge Context の繰り返しは禁止。"
        )

    if stateful:
        history.append({"role": "user", "content": user_text})
        prompt_history = history
    else:
        # Stateless stages: ignore session history entirely so prior
        # presenting/qa content can't bleed into the greeting/closing.
        prompt_history = [{"role": "user", "content": user_text}]

    # RAG retrieval — only meaningful in presenting / qa.
    # QA uses a smaller top_k so the [Knowledge Context] doesn't drown out
    # the audience's actual question in the model's attention. presenting
    # keeps the wider context since the slide alone often isn't enough.
    knowledge: list[dict] = []
    if req.stage in ("presenting", "qa"):
        query = _build_rag_query(req.stage, req.message, req.slide)
        if query:
            top_k = 2 if req.stage == "qa" else 5
            knowledge = retrieve_knowledge(query, top_k=top_k)

    # Per-response accuracy from the RAG hits we just gathered. Computed
    # here (before _record_qa) so both the persisted timeline entry and
    # the ChatResponse carry the same numbers without a second search.
    if knowledge:
        scores = [float(k.get("score", 0.0)) for k in knowledge]
        accuracy = {
            "rag_hits": len(scores),
            "top_score": max(scores) if scores else 0.0,
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
        }
    else:
        accuracy = {"rag_hits": 0, "top_score": 0.0, "mean_score": 0.0}

    slide_dict = req.slide.model_dump() if req.slide else None

    prompt_messages = build_prompt_messages(
        req.user_id,
        prompt_history,
        stage=req.stage,
        slide=slide_dict,
        knowledge=knowledge,
    )

    reply, truncated = generate_reply(prompt_messages)

    # TTS sanitizer: strip control chars, collapse whitespace.
    raw_reply = reply
    reply = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", reply)
    reply = re.sub(r"\s+", " ", reply).strip()

    # If truncated mid-sentence, trim to the last terminator so we don't
    # store a fragment that seeds an echo loop on the next turn.
    storable_reply = reply
    if truncated and reply:
        terminators = "。！？.!?"
        last_end = max((reply.rfind(c) for c in terminators), default=-1)
        if last_end >= 0:
            storable_reply = reply[: last_end + 1]
            reply = storable_reply
        else:
            storable_reply = ""
            logger.warning(
                "truncated reply with no sentence boundary for user=%s; not saving. raw=%r",
                req.user_id, raw_reply[:200],
            )

    # Persist only for stateful stages (presenting). waiting / qa / closing
    # leave the on-disk session untouched so each call stays independent.
    if stateful and not req.ephemeral:
        if storable_reply:
            history.append({"role": "assistant", "content": storable_reply})
        save_session(req.user_id, history)

    # Record this QA exchange to the global timeline so all participants can
    # see it on the right-hand panel. Runs only when there is a real question
    # and an answer was produced.
    if req.stage == "qa" and (req.message or "").strip() and storable_reply:
        _record_qa(
            user_id=req.user_id,
            question=req.message.strip(),
            answer=storable_reply,
            slide_page=req.slide.page if req.slide else None,
            lab_id=LAB_ID,
            accuracy=accuracy,
        )

    # Touch presence on any chat call so an active speaker counts as online.
    _touch_presence(req.user_id)

    return ChatResponse(
        user_id=req.user_id,
        reply=reply,
        stage=req.stage,
        slide_page=req.slide.page if req.slide else None,
        voice=(req.voice or "male"),
        lab_id=LAB_ID,
        accuracy=accuracy,
    )


@app.get("/api/config")
async def get_config():
    """Frontend bootstrap config. tts_url is a relative /api/ path so the
    browser stays on the proxy origin and CORS does not enter the picture."""
    return {"tts_url": TTS_PUBLIC_PATH}


def _build_lab_summary_payload() -> dict:
    """Compose the lab summary (RAG corpus stats + curator tagline + current LT meta).
    Used by the multi-lab fan-out flow: tunnel_client fetches this and posts it
    to the proxy via update_operator."""
    meta = None
    if _paper_rag is not None:
        try:
            meta = _paper_rag.get_index_meta()
        except Exception:
            logger.exception("paper_rag.get_index_meta failed")
    by_type = (meta or {}).get("by_type", {}) or {}
    papers_count = sum(by_type.get(k, 0) for k in ("paper", "preprint", "patent"))

    tagline = LAB_SUMMARY_TEXT.strip()
    if not tagline:
        bits = []
        theme = (current_meta.get("theme") or "").strip()
        if theme:
            bits.append(f"現在のテーマ「{theme}」")
        if papers_count:
            bits.append(f"論文 {papers_count} 本のRAG")
        chunks = (meta or {}).get("chunks")
        if chunks:
            bits.append(f"chunks={chunks}")
        tagline = "、".join(bits) if bits else "（summary 未設定）"
    if len(tagline) > LAB_SUMMARY_MAX_CHARS:
        tagline = tagline[: LAB_SUMMARY_MAX_CHARS - 1] + "…"

    return {
        "lab_id": LAB_ID,
        "name": LAB_NAME,
        "summary": tagline,
        "trust": {
            "corpus_chunks": (meta or {}).get("chunks", 0),
            "files": (meta or {}).get("files", 0),
            "papers": papers_count,
            "by_type": by_type,
            "last_updated": (meta or {}).get("built_at"),
            "embed_model": (meta or {}).get("model"),
            "embed_dim": (meta or {}).get("dim"),
        },
        "current_meta": dict(current_meta),
    }


@app.get("/api/lab/summary")
async def get_lab_summary():
    """Lab self-description for proxy fan-out / GUI trust-ranking.

    Fetched by tunnel_client after register and forwarded to the proxy as
    update_operator. No auth — this is public-ish lab metadata used by the
    multi-lab registry, not user data."""
    return _build_lab_summary_payload()


@app.post("/api/tts/generate_stream")
async def tts_generate_stream(req: Request, _uid: str = Depends(require_auth)):
    """Proxy raw-PCM streaming TTS from TTS_BACKEND to the browser.

    Body forwarded verbatim. Upstream X-Sample-Rate/Content-Type are passed
    through. Response is streamed chunk-by-chunk so playback can begin
    before the upstream finishes generation.
    """
    body = await req.body()
    headers = {"Content-Type": req.headers.get("Content-Type", "application/json")}
    upstream_url = f"{TTS_BACKEND.rstrip('/')}{TTS_BACKEND_PATH}"
    timeout = httpx.Timeout(TTS_TIMEOUT_S, connect=TTS_TIMEOUT_S, read=TTS_TIMEOUT_S)

    client = httpx.AsyncClient(timeout=timeout)
    try:
        stream_cm = client.stream("POST", upstream_url, content=body, headers=headers)
        upstream = await stream_cm.__aenter__()
    except Exception as e:
        await client.aclose()
        logger.exception("tts upstream connection failed")
        return JSONResponse({"error": f"tts upstream: {e}"}, status_code=502)

    if upstream.status_code != 200:
        err_body = await upstream.aread()
        await stream_cm.__aexit__(None, None, None)
        await client.aclose()
        logger.warning("tts upstream %d: %s", upstream.status_code, err_body[:200])
        return JSONResponse(
            {"error": f"tts upstream returned {upstream.status_code}"},
            status_code=502,
        )

    sample_rate = upstream.headers.get("X-Sample-Rate", "24000")
    content_type = upstream.headers.get("Content-Type", "application/octet-stream")

    async def relay():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await stream_cm.__aexit__(None, None, None)
            await client.aclose()

    return StreamingResponse(
        relay(),
        media_type=content_type,
        headers={"X-Sample-Rate": sample_rate},
    )


@app.post("/api/auth/google", response_model=GoogleAuthResponse)
async def auth_google(req: GoogleAuthRequest):
    """First-touch sign-in. Verifies the Google ID token, enforces the
    allowed-domain policy, and returns an existing user record (if any) or
    creates a new one."""
    try:
        info = _verify_google_credential(req.credential)
    except Exception as e:
        logger.warning("auth_google: invalid credential: %s", e)
        return JSONResponse({"detail": "invalid credential"}, status_code=401)

    email = (info.get("email") or req.email or "").lower()
    sub = info.get("sub") or req.sub
    if not email or not sub:
        return JSONResponse({"detail": "missing email/sub"}, status_code=400)
    if email != (req.email or "").lower() or sub != req.sub:
        return JSONResponse({"detail": "token claims do not match request"}, status_code=401)

    if not _is_allowed_email(email):
        logger.info("auth_google: domain rejected for %s", email)
        return JSONResponse(
            {"detail": f"domain not allowed (only {ALLOWED_EMAIL_DOMAIN})"},
            status_code=403,
        )

    users = _load_users()
    existing = next(
        (u for u in users.values() if u.get("sub") == sub),
        None,
    )
    if existing:
        if req.lang and existing.get("lang") != req.lang:
            existing["lang"] = req.lang
            _save_users(users)
        logger.info("auth_google: returning existing user_id=%s email=%s", existing["user_id"], email)
        return GoogleAuthResponse(
            userId=existing["user_id"],
            displayName=existing["display_name"],
            lang=existing.get("lang"),
        )

    new_id = str(uuid.uuid4())
    display_name = (
        (req.displayName or "").strip()
        or info.get("name")
        or email.split("@")[0]
    )
    users[new_id] = {
        "user_id": new_id,
        "email": email,
        "sub": sub,
        "display_name": display_name,
        "lang": req.lang,
        "created_at": int(time.time()),
    }
    _save_users(users)
    logger.info("auth_google: registered user_id=%s email=%s", new_id, email)
    return GoogleAuthResponse(userId=new_id, displayName=display_name, lang=req.lang)


@app.post("/api/auth/logout")
async def auth_logout(uid: str = Depends(require_auth)):
    """Sign the caller out. Removes them from the in-memory participants
    dict (with their qa entries). The persistent users.json registration is
    kept so re-sign-in stays seamless."""
    rec = participants.pop(uid, None)
    qa_count = len(rec.get("qa", [])) if rec else 0
    logger.info("[auth] logout user=%s removed=%s qa=%d", uid, bool(rec), qa_count)
    return {"status": "ok", "user_id": uid, "removed": bool(rec), "qa_dropped": qa_count}


@app.get("/api/auth/check")
async def auth_check(
    request: Request,
    lang: Optional[str] = None,
    difficulty: Optional[str] = None,  # accepted for compatibility; unused in LT
):
    """Verify an existing session by user_id (X-User-Id header)."""
    user_id = request.headers.get("X-User-Id") or request.headers.get("x-user-id")
    if not user_id:
        return JSONResponse({"detail": "no user"}, status_code=401)
    users = _load_users()
    user = users.get(user_id)
    if not user:
        return JSONResponse({"detail": "user not found"}, status_code=404)
    if lang and user.get("lang") != lang:
        user["lang"] = lang
        _save_users(users)
    return {
        "userId": user["user_id"],
        "displayName": user["display_name"],
        "lang": user.get("lang"),
    }


@app.post("/api/presence/heartbeat")
async def presence_heartbeat(req: HeartbeatRequest):
    """Browser pings every ~10s to declare 'I am here'. Updates last_seen.
    Records that survive disconnect remain in `participants` so the QA
    history they contributed stays visible."""
    if req.user_id not in _load_users():
        return JSONResponse({"detail": "unauthenticated"}, status_code=401)
    rec = _touch_presence(req.user_id)
    return {"status": "ok", "user_id": req.user_id, "last_seen": rec["last_seen"] if rec else None}


@app.get("/api/presence/users")
async def presence_users(_uid: str = Depends(require_auth)):
    """All known participants (Discord-like sidebar). Each entry has an
    `online` flag computed against PRESENCE_TIMEOUT_S, and `email_local`
    (the part before '@') for display."""
    auth_users = _load_users()
    rows = []
    for rec in participants.values():
        email = (auth_users.get(rec["user_id"]) or {}).get("email", "")
        email_local = email.split("@", 1)[0] if "@" in email else ""
        rows.append({
            "user_id": rec["user_id"],
            "display_name": rec["display_name"],
            "email_local": email_local,
            "joined_at": rec["joined_at"],
            "last_seen": rec["last_seen"],
            "online": _is_online(rec),
            "qa_count": len(rec.get("qa", [])),
        })
    rows.sort(key=lambda r: r["joined_at"])
    online_count = sum(1 for r in rows if r["online"])
    return {"count": len(rows), "online": online_count, "users": rows}


@app.get("/api/qa/timeline")
async def qa_timeline_get(
    start: int = 0,
    end: int = -1,
    limit: int = 10,
    _uid: str = Depends(require_auth),
):
    """Range query over QA entries by globally-unique qa_id.

    Semantics:
      start: lower bound, exclusive  (return entries with qa_id > start)
      end:   upper bound, inclusive  (-1 means no upper bound)
      limit: max items returned (oldest-within-range first, so the client
             can advance start to last item's qa_id on the next poll).

    Typical client usage:
      first poll  : start=0      &end=-1&limit=10
      next polls  : start={lastSeenId}&end=-1&limit=10
    """
    items = _all_qa_entries()
    if end == -1:
        items = [e for e in items if e["qa_id"] > start]
    else:
        items = [e for e in items if e["qa_id"] > start and e["qa_id"] <= end]
    items.sort(key=lambda e: e["qa_id"])
    if limit and len(items) > limit:
        items = items[:limit]
    return {
        "count": len(items),
        "max_qa_id": _qa_id_counter,
        "items": items,
    }


@app.put("/api/qa/{qa_id}")
async def qa_timeline_update(qa_id: int, edit: QAEditRequest, _uid: str = Depends(require_auth)):
    rec, idx = _find_qa(qa_id)
    if rec is None:
        return JSONResponse({"detail": "qa not found"}, status_code=404)
    entry = rec["qa"][idx]
    if edit.question is not None:
        entry["question"] = edit.question
    if edit.answer is not None:
        entry["answer"] = edit.answer
    entry["edited_at"] = time.time()
    return entry


@app.delete("/api/qa/{qa_id}")
async def qa_timeline_delete(qa_id: int, _uid: str = Depends(require_auth)):
    rec, idx = _find_qa(qa_id)
    if rec is None:
        return JSONResponse({"detail": "qa not found"}, status_code=404)
    removed = rec["qa"].pop(idx)
    return {"status": "deleted", "qa_id": qa_id, "removed": removed}


@app.post("/api/reset")
async def reset(user_id: str, _uid: str = Depends(require_auth)):
    if user_id in sessions:
        del sessions[user_id]
    path = _session_path(user_id)
    if path.exists():
        path.unlink()
    return {"status": "reset", "user_id": user_id}


@app.get("/api/deck")
async def get_deck():
    """Return the slide deck JSON the frontend uses to drive the talk."""
    deck_path = DATA_DIR / "deck.json"
    if not deck_path.exists():
        return JSONResponse({"error": "deck.json not found"}, status_code=404)
    try:
        return JSONResponse(json.loads(deck_path.read_text(encoding="utf-8")))
    except Exception as e:
        logger.exception("failed to read deck.json")
        return JSONResponse({"error": f"deck read error: {e}"}, status_code=500)


@app.get("/api/history")
async def get_history(user_id: str, _uid: str = Depends(require_auth)):
    if user_id in sessions:
        return JSONResponse(sessions[user_id])
    history = load_session(user_id)
    if history:
        return JSONResponse(history)
    return JSONResponse({"error": "session not found"}, status_code=404)


# =====================================================================
# Paper upload -> slide generation
# =====================================================================
#
# The browser-to-tunnel WebSocket caps frame size at 1 MB so PDFs cannot
# travel through the control plane. Instead we use a presigned-PUT detour:
#
#   1. POST /api/upload/presign {filename}
#        -> {job_id, key, url}   (server creates upload_jobs[job_id]
#           in state="awaiting_upload" and signs an S3 PUT URL).
#   2. Browser PUTs the PDF directly to S3 (CORS allows the CloudFront
#      origin). No proxy/tunnel involvement.
#   3. POST /api/upload/start {job_id}
#        -> {job_id, status}     (server downloads s3://bucket/key into
#           DATA_DIR/uploads/<job_id>/<filename>, then spawns
#           generate_slides.py).
#   4. Frontend polls GET /api/upload/jobs/{job_id} for status + log tail.
#   5. When status == "done", frontend loads
#      GET /api/upload/jobs/{job_id}/slides  (the generated single-page HTML).

_s3_client_cached = None


def _s3_client():
    global _s3_client_cached
    if _s3_client_cached is None:
        # SigV4 + virtual-host addressing keeps the presigned URL clean
        # and forward-compatible with private-bucket policies.
        _s3_client_cached = boto3.client(
            "s3",
            region_name=S3_UPLOAD_REGION,
            config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )
    return _s3_client_cached


def _upload_dir() -> Path:
    return DATA_DIR / "uploads"


def _safe_pdf_name(raw: Optional[str]) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(raw or "paper.pdf").name) or "paper.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


_TITLE_RE = re.compile(r"<title>\s*([^<]+?)\s*</title>", re.IGNORECASE | re.DOTALL)


def _extract_html_title(path: Path) -> Optional[str]:
    try:
        # Generated decks are small (<200 KB); a single read is fine.
        head = path.read_text(encoding="utf-8", errors="replace")[:8192]
    except Exception:
        return None
    m = _TITLE_RE.search(head)
    if not m:
        return None
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    return title or None


# generate_slides.py emits a `=== 論文メタ情報 ===` block at the top of
# the .essence.txt file with `key: value` lines. We pick the fields that
# drive the page header here. Stops at the next `=== ... ===` heading.
_ESSENCE_KEY_RE = re.compile(r"^\s*([^：:]+?)\s*[：:]\s*(.+?)\s*$")


def _parse_essence_meta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    meta: dict[str, str] = {}
    in_meta_block = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("===") and line.endswith("==="):
            label = line.strip("= ").strip()
            in_meta_block = ("論文メタ" in label) or ("メタ情報" in label)
            if not in_meta_block and meta:
                break
            continue
        if not in_meta_block or not line:
            continue
        m = _ESSENCE_KEY_RE.match(line)
        if not m:
            continue
        key = m.group(1).strip()
        val = m.group(2).strip().rstrip("，,").strip()
        # Drop the trailing "※..." annotation generate_slides.py sometimes adds.
        val = re.sub(r"\s*※.*$", "", val).strip()
        if key and val:
            meta[key] = val
    return meta


# generate_scripts.py emits sections in the form:
#   ## スライド 1: タイトル
#
#   発表スクリプト本文 (複数段落・改行を含む)
#
#   ---
#
#   ## スライド 2: タイトル
#   ...
# We parse it into [{index:int, title:str, lines:[str]}] where each
# non-empty line of the body becomes its own TTS unit.
_SCRIPT_HEADER_RE = re.compile(r"^##\s*スライド\s*(\d+)\s*[:：]\s*(.+?)\s*$")


def _parse_script_md(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    slides: list[dict] = []
    current: Optional[dict] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip() == "---":
            current = None
            continue
        m = _SCRIPT_HEADER_RE.match(line)
        if m:
            current = {
                "index": int(m.group(1)),
                "title": m.group(2).strip(),
                "lines": [],
            }
            slides.append(current)
            continue
        if current is None:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        current["lines"].append(stripped)
    slides.sort(key=lambda s: s["index"])
    return slides


def _meta_to_header(essence: dict[str, str]) -> dict[str, str]:
    """Project the parsed essence metadata onto the three header fields."""
    presenter = essence.get("著者") or essence.get("発表者") or ""
    # Author lines are often "杉澤洋祐，杉澤大輔" — normalize the comma.
    presenter = re.sub(r"[，,]\s*", " / ", presenter).strip()
    venue_parts = []
    for k in ("学会", "学会/研究会", "会議", "発表先"):
        if essence.get(k):
            venue_parts.append(essence[k])
            break
    if essence.get("所属") and not venue_parts:
        # Fall back to affiliation when no venue is given.
        venue_parts.append(essence["所属"])
    elif essence.get("所属"):
        venue_parts.append(essence["所属"])
    return {
        "presenter": presenter,
        "affiliation": essence.get("所属") or "",
        "venue": " · ".join(p for p in venue_parts if p),
    }


# =====================================================================
# RAG-augmented knowledge context for slide / script generation
# =====================================================================
# This is the "max" RAG path — it does substantially more than a single
# embedding lookup, because naive top-k cosine over a paper corpus mostly
# returns "documents that look like the input" (i.e., the same paper if
# it's already indexed, or its prior generated outputs). We instead:
#
#   1. LLM query rewriting: ask GPT to read the paper head and produce
#      THREE search queries from different angles (methods / domain /
#      evaluation), plus a one-line summary. This shifts the search
#      target from "papers like this one" to "papers we want to
#      contrast with this one".
#   2. Multi-query search: each rewritten query gets its own dense
#      lookup (paper_rag.search internally cross-encoder-reranks within
#      each query's candidate pool — RERANK_ENABLED is on by default).
#   3. Self-exclude: hits whose source/title contains the input PDF's
#      stem are dropped. This catches both the paper itself if it's
#      already in the corpus AND any prior `_presentation` /
#      `_presentation_script` derivatives.
#   4. Dedup + merge: same source across queries collapses to one row
#      keeping the max score across queries — items that score well from
#      multiple angles bubble to the top, items that only one query
#      liked still appear, and we get diversity for free.
#   5. Top-K final selection from the merged pool.
#
# If the LLM rewriting call fails or returns garbage we fall back to the
# previous behavior (single PDF-head query). Self-exclude + dedup still
# apply on the fallback path. RAG path is best-effort throughout: any
# error here makes the pipeline run knowledge-free instead of failing.

KNOWLEDGE_CONTEXT_QUERY_CHARS = int(os.environ.get("KNOWLEDGE_CONTEXT_QUERY_CHARS", "5000"))
KNOWLEDGE_CONTEXT_TOP_K = int(os.environ.get("KNOWLEDGE_CONTEXT_TOP_K", "8"))
# Per-query oversampling: each rewritten query pulls this many hits from
# paper_rag.search before self-exclude/dedup. Higher values give more
# room for diversity at the cost of one extra cross-encoder pass.
KNOWLEDGE_CONTEXT_PER_QUERY = int(os.environ.get("KNOWLEDGE_CONTEXT_PER_QUERY", "10"))
# Each chunk is truncated to this many chars before being written to the
# context file. Slide generation already feeds the full paper text into
# the same prompt, so we cap related-work chunks to keep the prompt
# budget bounded.
KNOWLEDGE_CONTEXT_CHUNK_CHARS = int(os.environ.get("KNOWLEDGE_CONTEXT_CHUNK_CHARS", "1200"))
# Model used for the query-rewriting call. Same family as the slide
# generator so terminology stays consistent.
KNOWLEDGE_REWRITER_MODEL = os.environ.get("KNOWLEDGE_REWRITER_MODEL", "gpt-5.2")
# Cap how much text we send to the rewriter. The PDF head can be 5KB+
# and we don't need that much for a query-design call — abstract +
# intro fit comfortably in 4KB.
KNOWLEDGE_REWRITER_INPUT_CHARS = int(os.environ.get("KNOWLEDGE_REWRITER_INPUT_CHARS", "4000"))
# Source-path substrings that, if found in a hit's source, cause it to
# be dropped from slide-generation knowledge context. Default kills the
# whole `professor_data/uploads/` tree because everything under it is
# either (a) a user-uploaded paper (= talk material, not external
# knowledge) or (b) a previously-generated artifact such as
# `presentation.html`, `presentation.essence.txt`, `_script.md`, or
# even `knowledge_context.md` from a prior run — feeding those back in
# creates a self-referential loop where the LLM cites its own past
# output as "prior art". Chat-time RAG retrieval (used during the actual
# talk) is intentionally NOT filtered by this — there the user's
# uploads ARE the relevant material.
KNOWLEDGE_CONTEXT_EXCLUDE_PREFIXES = [
    s.strip() for s in os.environ.get(
        "KNOWLEDGE_CONTEXT_EXCLUDE_PREFIXES",
        "professor_data/uploads",
    ).split(",") if s.strip()
]

# Curated external corpus directory for related-work PDFs. We mirror
# every uploaded paper into here right after it lands on disk so that
# the next RAG index rebuild (auto-fired at the end of each pipeline)
# picks it up — making it searchable as prior art for future uploads.
# Without this mirror, the only copy lives under professor_data/uploads/
# which is excluded from knowledge-context by KNOWLEDGE_CONTEXT_EXCLUDE_PREFIXES,
# so a paper uploaded on day 1 would never surface as related work for
# a paper uploaded on day 2 even though both are in the index.
EXTERNAL_PDF_DIR = Path(os.environ.get(
    "EXTERNAL_PDF_DIR",
    str(Path.home() / "git" / "paper" / "external-pdf-for-rag"),
))


def _stash_upload_to_external_corpus(job: dict, pdf_path: Path) -> Optional[Path]:
    """Copy the just-downloaded upload into the curated external corpus
    dir so the next index rebuild picks it up as prior art for future
    uploads. See EXTERNAL_PDF_DIR docstring for the rationale.

    Filename collisions overwrite intentionally — re-uploading the same
    paper should refresh the curated copy in place rather than pile up
    timestamped duplicates that would all surface as near-identical hits.
    Failure is non-fatal: the pipeline continues without the stash."""
    try:
        EXTERNAL_PDF_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        job["log"].append(f"[rag] external corpus mkdir failed: {e}; skipping stash")
        return None
    dest = EXTERNAL_PDF_DIR / pdf_path.name
    try:
        shutil.copy2(pdf_path, dest)
    except Exception as e:
        job["log"].append(f"[rag] external corpus copy failed: {e}")
        return None
    job["log"].append(
        f"[rag] stashed upload -> {dest} (next rebuild will index it as prior art)"
    )
    return dest


def _extract_pdf_head_text(pdf_path: Path, max_chars: int) -> str:
    """Pull the first max_chars of text from the PDF for use as a RAG query.
    Tries PyMuPDF first (matches generate_slides.py), then falls back to
    the lighter-weight pdfminer if PyMuPDF isn't installed. Returns "" on
    any failure — caller treats empty as "skip RAG, no context"."""
    try:
        import fitz  # type: ignore
        doc = fitz.open(str(pdf_path))
        try:
            buf: list[str] = []
            collected = 0
            for page in doc:
                t = page.get_text() or ""
                buf.append(t)
                collected += len(t)
                if collected >= max_chars:
                    break
            return "".join(buf)[:max_chars]
        finally:
            doc.close()
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        return (extract_text(str(pdf_path)) or "")[:max_chars]
    except Exception:
        logger.exception("knowledge-context: PDF head extraction failed for %s", pdf_path)
        return ""


def _load_openai_api_key() -> str:
    """Find an OpenAI API key for the rewriter call. Env wins; otherwise
    we read from the same .env the external slide-gen scripts use, since
    that's where the user already has it configured."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env_path = SLIDE_GEN_SCRIPT.parent / ".env"
    if not env_path.exists():
        return ""
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def _rewrite_queries_via_llm(pdf_head: str, job: dict) -> tuple[str, list[str]]:
    """Ask GPT to read the paper head and emit (summary, [q1, q2, q3]).

    The three queries target different perspectives so the resulting
    multi-search returns a diverse pool instead of three near-duplicates:

      q1: 手法/技術面の先行研究  (same category of methods)
      q2: 応用ドメインの関連研究 (application domain / use case)
      q3: 評価軸の関連研究        (comparison metrics, evaluation setup)

    Returns ("", []) on any failure — caller falls back to the single
    PDF-head query path. Job log gets a breadcrumb either way."""
    api_key = _load_openai_api_key()
    if not api_key:
        job["log"].append("[rag] rewriter: OPENAI_API_KEY not found; falling back to head query")
        return ("", [])
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        job["log"].append("[rag] rewriter: openai package unavailable; falling back to head query")
        return ("", [])

    head = re.sub(r"\s+", " ", pdf_head).strip()[:KNOWLEDGE_REWRITER_INPUT_CHARS]
    system_prompt = (
        "あなたは論文RAG検索のクエリ設計者です。"
        "新規論文の先行研究／関連研究との対比を引き出すため、"
        "観点の異なる検索クエリを **日本語3本 + 英語3本** の計6本で生成してください。"
        "RAG corpus には日本語論文と英語論文が混在しており、cross-lingual の embedding "
        "alignment が弱いため、両言語のクエリを並べないと片方の corpus にしか届かない。"
    )
    user_prompt = (
        "以下は新規論文の冒頭テキストです（先頭〜数千文字、欠落あり）。\n"
        "この論文を **既存研究と対比** するために RAG corpus を検索したい。\n"
        "観点 × 言語 の 3×2 = 6 クエリを設計してください。\n\n"
        "観点（共通）:\n"
        "  ・methods: 手法/技術面の先行研究 (同カテゴリの既存手法、アルゴリズム、機構)\n"
        "  ・domain : 応用ドメインの関連研究 (適用領域、ユースケース、ターゲット環境)\n"
        "  ・eval   : 評価軸/対比の関連研究 (比較指標、評価セットアップ、ベンチマーク)\n\n"
        "出力は JSON のみ、説明文・コードフェンス禁止:\n"
        "{\n"
        '  "summary": "論文の核心を一文で（日本語）",\n'
        '  "summary_en": "One-sentence summary (English).",\n'
        '  "queries_ja": ["methods 日本語", "domain 日本語", "eval 日本語"],\n'
        '  "queries_en": ["methods English",  "domain English",  "eval English"]\n'
        "}\n\n"
        "各クエリは 30〜80 文字程度の自然言語フレーズ。論文タイトル丸写しは禁止。\n"
        "英語クエリは英語論文 corpus に届くよう、英語の専門用語を主体にすること。\n\n"
        "--- 論文冒頭 ---\n"
        f"{head}\n"
        "--- ここまで ---"
    )
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=KNOWLEDGE_REWRITER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_completion_tokens=1500,
        )
    except Exception as e:
        job["log"].append(f"[rag] rewriter: LLM call failed ({e}); falling back to head query")
        return ("", [])

    msg = resp.choices[0].message
    content = msg.content
    # GPT-5.2 thinking-style content can be a list of parts.
    if isinstance(content, list):
        text = "".join(
            (p.text if hasattr(p, "text") else p.get("text", ""))
            for p in content
            if (getattr(p, "type", None) == "text"
                or (isinstance(p, dict) and p.get("type") == "text"))
        )
    else:
        text = content or ""
    text = text.strip()
    # Strip ```json fences if the model added them despite instructions.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except Exception:
        job["log"].append(
            f"[rag] rewriter: JSON parse failed; falling back to head query "
            f"(raw head: {text[:120]!r})"
        )
        return ("", [])
    summary = (data.get("summary") or "").strip()
    summary_en = (data.get("summary_en") or "").strip()
    # Accept both the new bilingual schema (queries_ja + queries_en) and
    # the legacy single-list schema (queries) so older deployments keep
    # working if the model regresses to old format. Bilingual takes
    # priority: combining both language pools is exactly what fixes the
    # cross-lingual embedding gap.
    queries_ja_raw = data.get("queries_ja") or []
    queries_en_raw = data.get("queries_en") or []
    queries_legacy_raw = data.get("queries") or []
    queries_ja = [str(q).strip() for q in queries_ja_raw if str(q).strip()]
    queries_en = [str(q).strip() for q in queries_en_raw if str(q).strip()]
    queries_legacy = [str(q).strip() for q in queries_legacy_raw if str(q).strip()]
    queries = queries_ja + queries_en if (queries_ja or queries_en) else queries_legacy
    if not queries:
        job["log"].append("[rag] rewriter: no queries in response; falling back to head query")
        return ("", [])
    job["log"].append(
        f"[rag] rewriter: {len(queries)} quer(y/ies) "
        f"({len(queries_ja)} ja + {len(queries_en)} en) "
        f"| summary={summary[:80]!r}"
    )
    if summary_en:
        job["log"].append(f"[rag] rewriter:   summary_en={summary_en[:80]!r}")
    for i, q in enumerate(queries, 1):
        lang = "ja" if i <= len(queries_ja) else "en"
        job["log"].append(f"[rag] rewriter:   q{i}({lang})={q!r}")
    return (summary, queries)


def _exclude_predicate(pdf_stem: str):
    """Return a predicate(doc) -> bool that's True when the doc should be
    dropped from the knowledge context. Two reasons we drop:

      1. Source/title contains the input paper's stem -> "self" hit
         (the paper itself or a derivative like `<stem>_presentation`).
      2. Source path contains any of KNOWLEDGE_CONTEXT_EXCLUDE_PREFIXES
         -> entire directory tree is off-limits. Default rules out
         everything under `professor_data/uploads/` because that path
         only contains user uploads + LLM-generated artifacts (which
         loop back on themselves if used as "prior art").

    Substring matching is intentional: paths in the index can be either
    absolute (`/home/.../professor_data/uploads/...`) or relative
    (`professor_data/uploads/...`), and stems like `time-locality-lfs.j`
    are also a substring of derivative names like
    `time-locality-lfs.j_presentation`, so substring covers both
    naming conventions."""
    needle_stem = pdf_stem.lower() if pdf_stem else ""
    needle_prefixes = [p.lower() for p in KNOWLEDGE_CONTEXT_EXCLUDE_PREFIXES]

    def should_exclude(doc: dict) -> bool:
        src = (doc.get("source") or "").lower()
        title = (doc.get("title") or "").lower()
        if needle_stem and (needle_stem in src or needle_stem in title):
            return True
        if any(p in src for p in needle_prefixes):
            return True
        return False

    return should_exclude


def _build_knowledge_context_file(job: dict, pdf_path: Path) -> Optional[Path]:
    """RAG-search the existing paper corpus for chunks related to this
    upload, write them as a markdown file in the job dir, and return the
    path. See module-level comment block for the full pipeline."""
    if _rag_search is None:
        job["log"].append("[rag] knowledge-context: paper_rag unavailable; skipping")
        return None
    head = _extract_pdf_head_text(pdf_path, KNOWLEDGE_CONTEXT_QUERY_CHARS)
    if not head.strip():
        job["log"].append("[rag] knowledge-context: PDF head empty; skipping")
        return None

    # Step 1: rewrite the PDF head into 3 perspective queries (with
    # graceful fallback to the raw head if rewriting fails).
    summary, queries = _rewrite_queries_via_llm(head, job)
    if not queries:
        # Fallback path: collapse whitespace and use the head as one
        # query. Self-exclude + dedup still apply below.
        queries = [re.sub(r"\s+", " ", head).strip()]

    # Step 2: per-query search. paper_rag.search internally reranks
    # within each query's candidate pool when PAPER_RAG_RERANK is on
    # (default). We oversample so the post-filter pool stays healthy.
    should_exclude = _exclude_predicate(pdf_path.stem)
    # source -> {"doc": dict, "score": float, "queries": [int]}
    merged: dict[str, dict] = {}
    skipped_excluded = 0
    raw_total = 0
    for qi, q in enumerate(queries, 1):
        try:
            hits = retrieve_knowledge(q, top_k=KNOWLEDGE_CONTEXT_PER_QUERY)
        except Exception:
            logger.exception("knowledge-context: search failed for q%d", qi)
            hits = []
        raw_total += len(hits)
        for doc in hits:
            if should_exclude(doc):
                skipped_excluded += 1
                continue
            # Use source path as the dedup key. If source is missing
            # (shouldn't happen for indexed docs) fall back to the first
            # 200 chars of text so we still collapse exact-text dupes.
            key = (doc.get("source") or "").strip() or (doc.get("text") or "")[:200]
            score = float(doc.get("score") or 0.0)
            cur = merged.get(key)
            if cur is None:
                merged[key] = {"doc": doc, "score": score, "queries": [qi]}
            else:
                cur["queries"].append(qi)
                if score > cur["score"]:
                    cur["score"] = score
                    cur["doc"] = doc  # prefer the higher-scoring variant

    job["log"].append(
        f"[rag] knowledge-context: {len(queries)} quer(y/ies), "
        f"raw {raw_total} hit(s), excluded {skipped_excluded} "
        f"(self+{KNOWLEDGE_CONTEXT_EXCLUDE_PREFIXES}), "
        f"deduped to {len(merged)}"
    )
    if not merged:
        job["log"].append("[rag] knowledge-context: nothing left after filtering; skipping")
        return None

    # Step 3: top-K from the merged pool, sorted by max score across
    # queries (ties broken by appearing in more queries — broader
    # relevance wins).
    ranked = sorted(
        merged.values(),
        key=lambda r: (r["score"], len(r["queries"])),
        reverse=True,
    )[:KNOWLEDGE_CONTEXT_TOP_K]

    out_lines: list[str] = []
    if summary:
        out_lines.append(f"_RAG クエリ書き換え summary_: {summary}")
        out_lines.append("")
    for i, row in enumerate(ranked, 1):
        doc = row["doc"]
        title = (doc.get("title") or "untitled").strip()
        source = (doc.get("source") or "").strip()
        text = (doc.get("text") or "").strip()
        if len(text) > KNOWLEDGE_CONTEXT_CHUNK_CHARS:
            text = text[:KNOWLEDGE_CONTEXT_CHUNK_CHARS].rstrip() + " …"
        # Show which perspective queries surfaced this hit so the
        # downstream LLM can weigh "endorsed by all 3 angles" higher
        # than "only methods angle liked it".
        q_marks = ",".join(f"q{i}" for i in row["queries"])
        out_lines.append(f"## [{i}] {title}  _(score={row['score']:.3f}, via {q_marks})_")
        if source:
            out_lines.append(f"_出典: {source}_")
        out_lines.append("")
        out_lines.append(text)
        out_lines.append("")

    # Always resolve to an absolute path: the subprocess runs with
    # cwd=<paper tools dir>, so a relative path here would mis-resolve
    # against that directory and the script would warn "not found".
    out_path = (pdf_path.parent / "knowledge_context.md").resolve()
    try:
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
    except Exception:
        logger.exception("knowledge-context: failed to write %s", out_path)
        job["log"].append(f"[rag] knowledge-context: write failed at {out_path}")
        return None
    job["log"].append(
        f"[rag] knowledge-context: {len(ranked)} hit(s) selected, "
        f"{out_path.stat().st_size} bytes -> {out_path.name}"
    )
    return out_path


async def _run_step(job: dict, label: str, argv: list[str], cwd: Path) -> int:
    """Run one pipeline step, mirroring stdout/stderr line-by-line into
    the job log so the modal can stream it. Returns the exit code."""
    job["log"].append(f"[step] {label}: {' '.join(argv)}")
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        job["log"].append(line.decode("utf-8", errors="replace").rstrip())
    return await proc.wait()


async def _run_slide_generation(job_id: str):
    job = upload_jobs.get(job_id)
    if not job:
        return

    pdf_path: Path = job["pdf_path"]
    # Resolve to an absolute path: the subprocess runs with cwd=script_dir
    # so a relative path here would mis-resolve against that directory.
    pdf_path_abs = pdf_path.resolve()
    job["status"] = "running"
    job["started_at"] = time.time()

    # The pipeline tools all live next to generate_slides.py and write
    # their outputs into that same dir. We run them with cwd=script_dir
    # so .env / node_modules / output paths line up.
    script_dir = SLIDE_GEN_SCRIPT.parent
    stem = pdf_path.stem  # e.g. "tech-jsample11"
    slides_stem = f"{stem}_presentation"  # used by html2pdf & generate_scripts

    gen_html = script_dir / f"{slides_stem}.html"
    gen_essence = script_dir / f"{slides_stem}.essence.txt"
    gen_pdf = script_dir / f"{slides_stem}.pdf"
    gen_script = script_dir / f"{slides_stem}_script.md"

    try:
        # Build the RAG-augmented knowledge context once, before either
        # generator runs. The same file is fed into generate_slides.py
        # (Step 1, essence + slide HTML prompts) and generate_scripts.py
        # (Step 3, speaker script prompt) so the related-work framing stays
        # consistent across both. The newly-uploaded paper itself is not
        # yet in the index — the index rebuild is enqueued only after this
        # whole pipeline succeeds — so there's no need to filter it out.
        knowledge_path = _build_knowledge_context_file(job, pdf_path)
        knowledge_args = (
            ["--knowledge-context", str(knowledge_path)] if knowledge_path else []
        )

        # Step 1: PDF -> HTML slides + essence
        rc = await _run_step(
            job, "generate_slides.py",
            [SLIDE_GEN_PYTHON, str(SLIDE_GEN_SCRIPT), str(pdf_path_abs), *knowledge_args],
            script_dir,
        )
        if rc != 0:
            job["status"] = "error"
            job["error"] = f"generate_slides.py exited with code {rc}"
            job["finished_at"] = time.time()
            return
        if not gen_html.exists():
            job["status"] = "error"
            job["error"] = f"output not found: {gen_html.name}"
            job["finished_at"] = time.time()
            return

        # Mirror outputs into the job dir so the rest of the system has a
        # stable path to read from.
        dest_html = pdf_path.parent / "presentation.html"
        shutil.copy2(gen_html, dest_html)
        job["html_path"] = dest_html
        if gen_essence.exists():
            dest_essence = pdf_path.parent / "presentation.essence.txt"
            shutil.copy2(gen_essence, dest_essence)
            job["essence_path"] = dest_essence

        # Commit the meta as soon as the slide HTML lands so the browser
        # header updates without waiting for html2pdf + script generation.
        theme = _extract_html_title(dest_html) or ""
        essence_meta = (_parse_essence_meta(job["essence_path"])
                        if job.get("essence_path") else {})
        header = _meta_to_header(essence_meta)
        meta_update = {
            "theme": theme,
            "presenter": header.get("presenter", ""),
            "affiliation": header.get("affiliation", ""),
            "venue": header.get("venue", ""),
        }
        job["theme"] = theme
        job["presenter"] = meta_update["presenter"]
        job["affiliation"] = meta_update["affiliation"]
        job["venue"] = meta_update["venue"]
        if theme or meta_update["presenter"] or meta_update["venue"]:
            global current_meta
            current_meta = meta_update
            job["log"].append(
                "[done] meta theme={!r} presenter={!r} affiliation={!r} venue={!r}".format(
                    meta_update["theme"], meta_update["presenter"],
                    meta_update["affiliation"], meta_update["venue"],
                )
            )
        job["log"].append(f"[done] html={dest_html.name} ({dest_html.stat().st_size} bytes)")

        # Step 2: HTML slides -> PDF (puppeteer/Chromium). The frontend
        # doesn't read the PDF directly; it's only the input to step 3.
        rc = await _run_step(
            job, "html2pdf.mjs",
            [SLIDE_NODE, str(SLIDE_HTML2PDF_SCRIPT), str(gen_html)],
            script_dir,
        )
        if rc != 0:
            job["status"] = "error"
            job["error"] = f"html2pdf.mjs exited with code {rc}"
            job["finished_at"] = time.time()
            return
        if not gen_pdf.exists():
            job["status"] = "error"
            job["error"] = f"slides pdf not found: {gen_pdf.name}"
            job["finished_at"] = time.time()
            return
        dest_pdf = pdf_path.parent / "presentation.pdf"
        shutil.copy2(gen_pdf, dest_pdf)
        job["pdf_slides_path"] = dest_pdf
        job["log"].append(f"[done] pdf={dest_pdf.name} ({dest_pdf.stat().st_size} bytes)")

        # Step 3: paper.pdf + slides.pdf -> per-slide speaker script.
        # Same knowledge_args as Step 1 so the speaker can verbally
        # reference the related work shown on the slides.
        rc = await _run_step(
            job, "generate_scripts.py",
            [SLIDE_GEN_PYTHON, str(SLIDE_SCRIPT_GEN_SCRIPT),
             str(pdf_path_abs), str(gen_pdf), *knowledge_args],
            script_dir,
        )
        if rc != 0:
            job["status"] = "error"
            job["error"] = f"generate_scripts.py exited with code {rc}"
            job["finished_at"] = time.time()
            return
        if not gen_script.exists():
            job["status"] = "error"
            job["error"] = f"script.md not found: {gen_script.name}"
            job["finished_at"] = time.time()
            return
        dest_script = pdf_path.parent / "presentation_script.md"
        shutil.copy2(gen_script, dest_script)
        job["script_path"] = dest_script
        # Pre-parse the script per slide so the API can return JSON shape.
        job["script_slides"] = _parse_script_md(dest_script)
        job["log"].append(
            f"[done] script={dest_script.name} ({dest_script.stat().st_size} bytes,"
            f" {len(job['script_slides'])} slides)"
        )

        job["status"] = "done"
        job["finished_at"] = time.time()

        # Enqueue a RAG rebuild so the just-uploaded paper becomes
        # searchable in QA. The single FIFO worker (_rag_rebuild_worker)
        # processes triggers in arrival order — one rebuild + atomic
        # swap per upload, no concurrency between rebuilds.
        _enqueue_rag_rebuild(f"job:{job['job_id']}")
    except Exception as e:
        logger.exception("slide generation failed")
        job["status"] = "error"
        job["error"] = str(e)
        job["finished_at"] = time.time()
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except Exception:
                pass


@app.post("/api/upload/presign")
async def upload_presign(req: UploadPresignRequest, uid: str = Depends(require_ticket)):
    if not SLIDE_GEN_SCRIPT.exists():
        raise HTTPException(500, f"slide generator missing: {SLIDE_GEN_SCRIPT}")

    safe_name = _safe_pdf_name(req.filename)
    job_id = uuid.uuid4().hex[:12]
    key = f"{S3_UPLOAD_PREFIX.lstrip('/')}{job_id}/{safe_name}"

    try:
        url = _s3_client().generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": S3_UPLOAD_BUCKET,
                "Key": key,
                "ContentType": "application/pdf",
            },
            ExpiresIn=S3_PRESIGN_TTL_S,
            HttpMethod="PUT",
        )
    except Exception as e:
        logger.exception("presign failed")
        raise HTTPException(500, f"presign failed: {e}")

    upload_jobs[job_id] = {
        "job_id": job_id,
        "user_id": uid,
        "filename": safe_name,
        "status": "awaiting_upload",
        "log": [f"[presign] key={key}"],
        "started_at": None,
        "finished_at": None,
        "pdf_path": None,
        "s3_bucket": S3_UPLOAD_BUCKET,
        "s3_key": key,
        "html_path": None,
        "essence_path": None,
        "error": None,
    }
    return {"job_id": job_id, "key": key, "url": url, "filename": safe_name}


async def _download_from_s3(job_id: str):
    """Pull the uploaded PDF from S3 into the local job dir, then kick the
    slide generator. Runs as a background task so /api/upload/start returns
    immediately."""
    job = upload_jobs.get(job_id)
    if not job:
        return
    bucket = job["s3_bucket"]
    key = job["s3_key"]
    safe_name = job["filename"]

    job["status"] = "downloading"
    job["log"].append(f"[s3] get s3://{bucket}/{key}")
    job_dir = _upload_dir() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = job_dir / safe_name

    try:
        max_bytes = UPLOAD_PDF_MAX_MB * 1024 * 1024
        # HEAD first so we can reject oversize PDFs before pulling them down.
        head = await asyncio.to_thread(
            _s3_client().head_object, Bucket=bucket, Key=key,
        )
        size = int(head.get("ContentLength", 0))
        if size > max_bytes:
            shutil.rmtree(job_dir, ignore_errors=True)
            job["status"] = "error"
            job["error"] = f"pdf exceeds {UPLOAD_PDF_MAX_MB} MB ({size} bytes)"
            job["finished_at"] = time.time()
            return
        await asyncio.to_thread(
            _s3_client().download_file, bucket, key, str(pdf_path),
        )
        job["pdf_path"] = pdf_path
        job["log"].append(f"[s3] downloaded {safe_name} ({size} bytes)")
        # Mirror into the curated external corpus dir BEFORE the
        # pipeline runs — the auto-rebuild fired at the end of
        # _run_slide_generation will then include it.
        _stash_upload_to_external_corpus(job, pdf_path)
    except Exception as e:
        logger.exception("s3 download failed")
        job["status"] = "error"
        job["error"] = f"s3 download failed: {e}"
        job["finished_at"] = time.time()
        return

    await _run_slide_generation(job_id)

    # Drop the S3 copy after the pipeline runs (success or fail) — we have
    # the local copy under DATA_DIR/uploads/ for any post-mortem.
    try:
        await asyncio.to_thread(
            _s3_client().delete_object, Bucket=bucket, Key=key,
        )
    except Exception:
        logger.warning("s3 delete failed for %s/%s", bucket, key)


@app.get("/api/upload/eligibility")
async def upload_eligibility(uid: str = Depends(require_auth)):
    """Frontend uses this to decide which paid UI to show. Returns one
    eligibility flag per ticket action (currently `upload` and
    `remove`) plus the email checked, so the UI can show/hide buttons
    independently and surface a meaningful "ticket required" hint.

    Top-level `eligible` is kept as an alias of `upload` for backward
    compat with the previous single-flag response shape."""
    upload_ok = _user_has_ticket(uid, "upload")
    remove_ok = _user_has_ticket(uid, "remove")
    return {
        "eligible": upload_ok,           # legacy alias
        "upload": upload_ok,
        "remove": remove_ok,
        "email": _user_email(uid),
    }


@app.post("/api/upload/start")
async def upload_start(req: UploadStartRequest, uid: str = Depends(require_ticket)):
    job = upload_jobs.get(req.job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] != "awaiting_upload":
        raise HTTPException(409, f"job is in state={job['status']}")
    # Single-use ticket: consume one .ticket.available -> .ticket.consumed
    # before kicking off the pipeline. require_ticket already verified
    # *some* ticket existed at request entry, but that's a soft check —
    # we commit by atomic rename here, which also handles the race
    # between two concurrent /start calls fighting for the last ticket.
    email = _user_email(uid)
    consumed = _consume_ticket_for(email, "upload")
    if consumed is None:
        raise HTTPException(
            403,
            f"no ticket: アップロード用チケット消費失敗（残数0 or 別リクエストが先取り） email={email}",
        )
    job["log"].append(f"[ticket] consumed {consumed.name} (email={email}, action=upload)")
    job["ticket_path"] = str(consumed)
    job["status"] = "queued"
    asyncio.create_task(_download_from_s3(req.job_id))
    return {"job_id": req.job_id, "status": "queued"}


@app.get("/api/upload/jobs/{job_id}")
async def upload_job_status(
    job_id: str,
    since: int = 0,
    _uid: str = Depends(require_auth),
):
    job = upload_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    log_list = job["log"]
    total = len(log_list)
    # ?since=<offset> lets the frontend pull only the lines it hasn't
    # seen yet, so polling stays cheap even during multi-minute runs.
    start = max(0, min(since, total))
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "filename": job["filename"],
        "log": log_list[start:],
        "log_offset": start,
        "log_total": total,
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "error": job["error"],
        "html_ready": job["html_path"] is not None,
        "script_ready": bool(job.get("script_slides")),
        "theme": job.get("theme") or "",
        "presenter": job.get("presenter") or "",
        "venue": job.get("venue") or "",
    }


@app.get("/api/upload/jobs/{job_id}/script")
async def upload_job_script(job_id: str, _uid: str = Depends(require_auth)):
    job = upload_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    slides = job.get("script_slides")
    if not slides:
        raise HTTPException(409, f"script not ready (status={job['status']})")
    return {"slides": slides}


def _rehydrate_papers_from_disk() -> int:
    """Walk professor_data/uploads/* and recreate upload_jobs entries for
    every directory that has the full set of artifacts (HTML + script.md).
    Lets users pick a previously-processed paper across server restarts."""
    base = _upload_dir()
    if not base.exists():
        return 0
    n = 0
    for jd in sorted(base.iterdir()):
        if not jd.is_dir():
            continue
        job_id = jd.name
        if job_id in upload_jobs:
            continue
        html = jd / "presentation.html"
        script = jd / "presentation_script.md"
        if not (html.exists() and script.exists()):
            continue
        # The original paper PDF is whichever .pdf is NOT the slide deck.
        pdfs = [p for p in jd.glob("*.pdf") if p.name != "presentation.pdf"]
        if not pdfs:
            continue
        pdf_path = pdfs[0]
        essence = jd / "presentation.essence.txt"
        slides_pdf = jd / "presentation.pdf"
        theme = _extract_html_title(html) or ""
        essence_meta = _parse_essence_meta(essence) if essence.exists() else {}
        header = _meta_to_header(essence_meta)
        upload_jobs[job_id] = {
            "job_id": job_id,
            "user_id": None,             # rehydrated, original uploader unknown
            "filename": pdf_path.name,
            "status": "done",
            "log": [f"[rehydrate] {jd}"],
            "started_at": None,
            "finished_at": jd.stat().st_mtime,
            "pdf_path": pdf_path,
            "s3_bucket": None,
            "s3_key": None,
            "html_path": html,
            "essence_path": essence if essence.exists() else None,
            "pdf_slides_path": slides_pdf if slides_pdf.exists() else None,
            "script_path": script,
            "script_slides": _parse_script_md(script),
            "theme": theme,
            "presenter": header.get("presenter", ""),
            "affiliation": header.get("affiliation", ""),
            "venue": header.get("venue", ""),
            "error": None,
        }
        n += 1
    return n


@app.get("/api/upload/papers")
async def list_uploaded_papers(_uid: str = Depends(require_auth)):
    """List every uploaded paper that has a complete artifact set
    (HTML deck + per-slide script). Sorted newest first."""
    items = []
    for job_id, job in upload_jobs.items():
        if job.get("status") != "done":
            continue
        if not job.get("html_path") or not job.get("script_slides"):
            continue
        items.append({
            "job_id": job_id,
            "filename": job.get("filename") or "",
            "theme": job.get("theme") or "",
            "presenter": job.get("presenter") or "",
            "affiliation": job.get("affiliation") or "",
            "venue": job.get("venue") or "",
            "modified_at": job.get("finished_at"),
            "slides": len(job.get("script_slides") or []),
        })
    items.sort(key=lambda x: x["modified_at"] or 0, reverse=True)
    return {"papers": items}


@app.delete("/api/upload/papers/{job_id}")
async def delete_uploaded_paper(job_id: str, uid: str = Depends(require_remove_ticket)):
    """Drop a previously-uploaded paper from the server: remove its
    on-disk artifacts and forget the in-memory job entry. If this paper
    was the active talk, clear current_meta so the header resets.

    Single-use remove ticket is consumed before any destructive action,
    so a 403 (ticket race) never leaves orphan files behind."""
    job = upload_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "paper not found")
    email = _user_email(uid)
    consumed = _consume_ticket_for(email, "remove")
    if consumed is None:
        raise HTTPException(
            403,
            f"no ticket: 削除用チケット消費失敗（残数0 or 別リクエストが先取り） email={email}",
        )
    logger.info("delete: consumed %s (email=%s) for job %s", consumed.name, email, job_id)
    job_dir = _upload_dir() / job_id
    try:
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
    except Exception as e:
        logger.exception("failed to remove %s", job_dir)
        raise HTTPException(500, f"delete failed: {e}")
    upload_jobs.pop(job_id, None)
    # If the deleted paper was driving the active meta, clear it.
    global current_meta
    if (current_meta.get("theme") == (job.get("theme") or "")
            and current_meta.get("presenter") == (job.get("presenter") or "")
            and current_meta.get("venue") == (job.get("venue") or "")
            and current_meta.get("theme")):
        current_meta = {"theme": "", "presenter": "", "affiliation": "", "venue": ""}
    return {"job_id": job_id, "deleted": True}


@app.post("/api/upload/papers/{job_id}/select")
async def select_uploaded_paper(job_id: str, _uid: str = Depends(require_auth)):
    """Make a previously-uploaded paper the active talk: refresh
    current_meta so /api/chat sees the new theme/persona context.
    The browser still calls /api/upload/jobs/{id}/slides and /script
    afterwards to pull the artifacts in."""
    job = upload_jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "paper not ready")
    global current_meta
    current_meta = {
        "theme": job.get("theme") or "",
        "presenter": job.get("presenter") or "",
        "affiliation": job.get("affiliation") or "",
        "venue": job.get("venue") or "",
    }
    return {
        "job_id": job_id,
        "filename": job.get("filename") or "",
        "theme": current_meta["theme"],
        "presenter": current_meta["presenter"],
        "affiliation": current_meta["affiliation"],
        "venue": current_meta["venue"],
    }


@app.get("/api/meta")
async def get_meta(_uid: str = Depends(require_auth)):
    return dict(current_meta)


# =====================================================================
# Ticket CRUD  (paper upload / paper delete tickets, ticket-admin only)
# =====================================================================
# Tickets are single-use credit files placed under TICKETS_DIR; they
# gate /api/upload/start (action="upload") and DELETE /api/upload/papers
# (action="remove"). This CRUD lets a ticket-admin mint, list, edit,
# and revoke them without shelling into the host. All five endpoints
# require the "ticket-admin" role granted via CUSTOM_AUTH_FILE.

class TicketCreateRequest(BaseModel):
    email: str
    action: str  # "upload" | "remove"
    note: Optional[str] = ""
    purchased_at: Optional[str] = ""
    transaction_id: Optional[str] = ""


class TicketUpdateRequest(BaseModel):
    email: Optional[str] = None
    note: Optional[str] = None
    purchased_at: Optional[str] = None
    transaction_id: Optional[str] = None


require_ticket_admin = require_role("ticket-admin")


@app.get("/api/ticket/")
async def list_tickets(
    action: Optional[str] = None,
    status: Optional[str] = None,
    _uid: str = Depends(require_ticket_admin),
):
    """List every ticket file. Optional ?action=upload|remove and
    ?status=available|consumed filters narrow the view. Sorted with
    available tickets first (oldest → newest), consumed last."""
    items = [_ticket_view(p) for p in _list_all_ticket_paths()]
    if action:
        items = [t for t in items if t["action"] == action]
    if status:
        items = [t for t in items if t["status"] == status]
    items.sort(key=lambda t: (t["status"] != "available", t["created_at"]))
    return {"tickets": items}


@app.post("/api/ticket/", status_code=201)
async def create_ticket(
    req: TicketCreateRequest,
    uid: str = Depends(require_ticket_admin),
):
    """Mint a new available ticket. Writes
    TICKETS_DIR/<uuid>.ticket.<suffix> with action+email+metadata in
    the JSON so consumed tickets remain identifiable post-rename."""
    if req.action not in TICKET_ACTION_SUFFIX:
        raise HTTPException(
            400, f"invalid action: {req.action!r} (expected one of {list(TICKET_ACTION_SUFFIX)})",
        )
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "invalid email")
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    ticket_id = str(uuid.uuid4())
    suffix = TICKET_ACTION_SUFFIX[req.action]
    path = TICKETS_DIR / f"{ticket_id}{suffix}"
    payload = {
        "ticket_id": ticket_id,
        "email": email,
        "action": req.action,
        "note": req.note or "",
        "purchased_at": req.purchased_at or "",
        "transaction_id": req.transaction_id or "",
        "created_at": int(time.time()),
        "created_by": _user_email(uid) or uid,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info(
        "ticket: created %s for %s by %s", path.name, email, payload["created_by"],
    )
    return _ticket_view(path)


@app.get("/api/ticket/{ticket_id}")
async def get_ticket(
    ticket_id: str,
    _uid: str = Depends(require_ticket_admin),
):
    p = _ticket_path_for(ticket_id)
    if p is None:
        raise HTTPException(404, "ticket not found")
    return _ticket_view(p)


@app.put("/api/ticket/{ticket_id}")
async def update_ticket(
    ticket_id: str,
    req: TicketUpdateRequest,
    uid: str = Depends(require_ticket_admin),
):
    """Edit an existing ticket's email / note / receipt metadata.
    Action and status are immutable here — to change them, delete and
    re-create."""
    p = _ticket_path_for(ticket_id)
    if p is None:
        raise HTTPException(404, "ticket not found")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if req.email is not None:
        e = req.email.strip().lower()
        if not e or "@" not in e:
            raise HTTPException(400, "invalid email")
        data["email"] = e
        # If the ticket was authored under the legacy `google_account`
        # field, drop it so future reads don't see two sources of truth.
        data.pop("google_account", None)
    if req.note is not None:
        data["note"] = req.note
    if req.purchased_at is not None:
        data["purchased_at"] = req.purchased_at
    if req.transaction_id is not None:
        data["transaction_id"] = req.transaction_id
    data["updated_at"] = int(time.time())
    data["updated_by"] = _user_email(uid) or uid
    p.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info("ticket: updated %s by %s", p.name, data["updated_by"])
    return _ticket_view(p)


@app.delete("/api/ticket/{ticket_id}", status_code=204)
async def delete_ticket(
    ticket_id: str,
    uid: str = Depends(require_ticket_admin),
):
    """Hard-delete a ticket file (any status). Use with care: deleting
    a consumed ticket erases its audit trail."""
    p = _ticket_path_for(ticket_id)
    if p is None:
        raise HTTPException(404, "ticket not found")
    try:
        p.unlink()
    except Exception as e:
        logger.exception("ticket: delete failed for %s", p)
        raise HTTPException(500, f"delete failed: {e}")
    logger.info("ticket: deleted %s by %s", p.name, _user_email(uid) or uid)
    # 204 must have an empty body — use Response (not JSONResponse, which
    # would serialize None to "null" and trip uvicorn's Content-Length
    # enforcement).
    return Response(status_code=204)


@app.get("/api/upload/jobs/{job_id}/slides", response_class=HTMLResponse)
async def upload_job_slides(job_id: str, _uid: str = Depends(require_auth)):
    job = upload_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] != "done" or not job["html_path"]:
        raise HTTPException(409, f"slides not ready (status={job['status']})")
    return FileResponse(job["html_path"], media_type="text/html; charset=utf-8")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>D.Sugisawa LT</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column;
  }
  header { text-align: center; padding: 16px; background: rgba(0,0,0,0.3); border-bottom: 1px solid #e94560; }
  header h1 { font-size: 1.2em; color: #e94560; margin-bottom: 4px; line-height: 1.4; }
  header p { font-size: 0.85em; color: #aaa; }
  main { flex: 1; padding: 16px; max-width: 960px; margin: 0 auto; width: 100%; }
  #status { padding: 14px; background: #1e2a4a; border-radius: 8px; margin-bottom: 12px; line-height: 1.6; }
  #log { padding: 12px; background: #11192e; border-radius: 8px; min-height: 240px; max-height: 60vh; overflow-y: auto; font-size: 0.95em; line-height: 1.6; }
  .turn { margin-bottom: 8px; padding: 6px 8px; border-radius: 6px; }
  .turn.user { background: #2a1a2e; }
  .turn.assistant { background: #1a2a4a; }
  .meta { color: #888; font-size: 0.78em; margin-bottom: 2px; }
</style>
</head>
<body>
<header>
  <h1 id="talk-title">= = = =</h1>
  <p>= = = =</p>
</header>
<main>
  <div id="status">準備中… まもなく開始します。</div>
  <div id="log"></div>
</main>
<script>
const params = new URLSearchParams(location.search);
const userId = params.get('user_id') || crypto.randomUUID();
const log = document.getElementById('log');
const statusEl = document.getElementById('status');

function appendTurn(role, text, meta) {
  const d = document.createElement('div');
  d.className = 'turn ' + role;
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    d.appendChild(m);
  }
  const t = document.createElement('div');
  t.textContent = text;
  d.appendChild(t);
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

async function callChat(payload) {
  const res = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_id: userId, ...payload}),
  });
  return res.json();
}

// Trigger a 'waiting' utterance on load so the audience hears the intro
(async () => {
  try {
    const data = await callChat({stage: 'waiting'});
    appendTurn('assistant', data.reply, 'stage=waiting');
    statusEl.textContent = data.reply;
  } catch (e) {
    appendTurn('assistant', '接続エラー: ' + e.message);
  }
})();
</script>
</body>
</html>
"""


def main():
    global generator, encoding

    parser = argparse.ArgumentParser(description="Professor LT HTTP Server")
    parser.add_argument("--checkpoint", default="openai/gpt-oss-20b", help="Model name or path")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--data-dir", default="professor_data", help="Directory for persistent data")
    parser.add_argument("--context-length", type=int, default=16384,
                        help="Max model context length passed to the vLLM engine")
    args = parser.parse_args()

    global DATA_DIR, SESSIONS_DIR
    DATA_DIR = Path(args.data_dir)
    SESSIONS_DIR = DATA_DIR / "sessions"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
    print(f"Data directory ready: {DATA_DIR}")

    from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator

    print(f"Loading model: {args.checkpoint} (vLLM backend, max_model_len={args.context_length}) ...")
    generator = VLLMGenerator(
        args.checkpoint,
        tensor_parallel_size=1,
        max_model_len=args.context_length,
    )
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    print("Model loaded with vLLM backend.")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
