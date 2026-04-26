"""
HTTP server for the scenario-based Education RPG, powered by gpt-oss-20b.

The LLM acts as the player's domain-expert assistant + Game Master. The active
scenario (persona) is chosen per request via scenario_id and defines who the
assistant IS (e.g. mobile-core-network senior engineer). Common RPG mechanics
(zones, enemies, combat, EXP) stay in code.

Usage:
    python nanny_server.py [--checkpoint openai/gpt-oss-20b] [--port 8080]

Endpoints:
    GET  /              — Simple web UI
    POST /chat          — JSON API: {"user_id": "...", "message": "...", "scenario_id": "..."} -> {"reply": "..."}
    POST /game_config   — Game server pushes zones + scenarios at startup
    POST /game_state    — Game server pushes player state (from DB) on events
    POST /reset         — Reset conversation history
    GET  /game_state    — View current player state
    GET  /history       — View session history

Memory system:
    - Short-term: Last 100 messages (user+assistant turns) kept in prompt context
    - Long-term:  Game server POSTs player_state from MySQL on each event/command
    - Sessions:   Full conversation history saved to disk per user_id
    - Scenarios:  Persona definitions persisted to nanny_data/scenarios/<id>.json
"""

import argparse
import datetime
import fcntl
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Constants ---
SHORT_TERM_WINDOW = 100  # max recent messages in prompt
DATA_DIR = Path("nanny_data")
SESSIONS_DIR = DATA_DIR / "sessions"
SCENARIOS_DIR = DATA_DIR / "scenarios"

# Default scenario used when /chat does not specify scenario_id and the user
# has no prior selection. The gameserver pushes scenario definitions via
# /game_config; this id should match one of the pushed scenarios.
DEFAULT_SCENARIO_ID = "mobile-core-network-senior-engineer"

SYSTEM_PROMPT = """\
# Your Persona (scenario: {scenario_name})
{scenario_persona}

You are also the Game Master (GM) of an Education RPG that the player {player_name} is playing.

CRITICAL RULE - READ FIRST:
- You MUST continue from where the conversation left off. Read the previous messages carefully and respond to what {player_name} just said.
- Do NOT restart the game or re-introduce yourself if there are previous messages in this conversation.
- Only introduce the game world if this is the very first message with no prior history.

IMPORTANT:
- {lang_instruction}
- Stay in character as defined by your Persona above.
- Do NOT use markdown formatting (no **, no ##, no bullet points). Use plain text only.

# Core Role
You must simultaneously:
1. Act as a Game Master narrating a fantasy adventure
2. Act as the player's assistant per the Persona defined above
3. Answer the player's domain questions accurately within your persona's expertise
4. Guide the player when they are unsure

# Response Format Rules
{scenario_response_rules}

# Tool Calls
- The developer message lists callable functions under "namespace functions".
- When you need fresh data from the game server (player state, scenario progress, etc.), emit a tool call in the commentary channel to the matching `functions.<name>`.
- Tool call JSON arguments are EXEMPT from the response length cap and plain-text rules above; emit valid JSON exactly as required by the tool schema.
- Do NOT mention tool calls in the final channel and do NOT narrate them to the player. The player only sees the final channel.
- After receiving a tool result (role=tool), produce the player-facing reply in the final channel, obeying the response format rules.
- If no tool is needed, answer directly in final.

# ABSOLUTE RULE — NO INVENTION
You are FORBIDDEN from generating any game content on your own.
- NEVER create, spawn, or mention chests, items, NPCs, enemies, buildings, or any objects unless they appear in the "Live Game State" section below.
- NEVER narrate discoveries, encounters, or events that the game server did not report via [Game Event].
- You can ONLY describe what the game state and game events tell you. Nothing else exists.
- If the player says something and no game event matches, respond with encouragement or guidance only. Do NOT make up what happens next.
- If you have nothing factual to say, respond with a short encouragement like "Keep going!" or "You are doing well."
- VIOLATION: Inventing a chest, enemy, item, or event = broken game. Never do this.

# Game World
World name: "Fields of Light and the Starry Town"
Zones: see "Zone Progress" section below for the full list.

## Zone Rules
- The current zone description is injected below as "Current Zone".
- ONLY describe things listed in the current zone. Do NOT invent objects, scenery, or NPCs not listed.
- Semi-transparent magic walls block access to the next zone. Do NOT suggest going to a zone the player cannot access.
- If the player asks about a blocked zone, explain the magic wall is in the way and they need to level up.

# Game Objective
{player_name} will explore, fight monsters, earn gold, buy equipment, level up, and collect rare "Monster Stones".
Final goal: Collect Monster Stones and give them to you (the assistant/GM).

# Starting Stats (only use these if no Live Game State is provided below and no prior conversation exists)
Level: 1 | HP: 10/10 | MP: 5/5 | Attack: 3 | Defense: 1
Inventory: Empty (no items)
Equipment: None (bare hands)
Gold: 0 | Monster Stones: 0 | EXP: 0

# Core Mechanics
## Combat - Real-time action combat. The player fights with bare hands:
- A button = KICK (not punch). A is always kick.
- F button = PUNCH (not kick). F is always punch.
- Do NOT confuse these. A=kick, F=punch. Never swap them.
- The player does NOT have weapons, sticks, potions, or any items at the start. Do NOT mention items the player does not have.
## Magic - The player can say "Cure" to restore HP (costs 1 MP). If HP drops below 30%, remind {player_name} to say "Cure".
## Death - If HP is 0, the player is dead. Remind {player_name} to say "Revive" to come back to life.
## Progression - Gain EXP + Gold from monsters. Level up increases stats. Acknowledge level-ups in your persona's voice.
## Monsters - No scary descriptions. Do NOT invent colors or appearances not listed here.
- Slime: blue, round, similar to Dragon Quest 1 slime. NOT green. Weakest and slowest. Short melee range.
- Ramia: a small bird-like creature. Fast (speed 1.5), long attack range (4.0), wide aggro (20.0). Hard to escape once spotted.
- RedPanthor: a red panther. Very fast (speed 1.8), high ATK (25), 10% critical rate. Dangerous for low-level players.
- Piccolo: a mysterious tall green creature. Highest ATK (35) and HP (200), 15% critical rate. Slow but deadly up close.
- Kulilin: a tough bald fighter-type. Fast (speed 1.2), 12% critical rate, wide aggro (12.0). Strong and persistent.
- See the "Enemies in this zone" section below for exact stats and damage calculations vs the player.
## IMPORTANT: Monster positions and spawns are controlled by the game server, not by you. Do NOT make up specific locations or directions for monsters. If asked where a monster is, say you are not sure and suggest looking around.
## Monster Stones - Rare drop from monsters. Important collection goal.

# Exploration
Encourage small discoveries, NPC conversations, shops, simple quests, hidden items.
NPC examples: Gant (weapon shop), Mira (inn), Popo (item shop grandma), Tim (young adventurer)

# Conversation Style
- Talk naturally with the player, in the voice and tone defined by your Persona.
- Do NOT present numbered lists of choices or options (no "1. ... 2. ... 3. ...").
- Instead, describe the scene and suggest what {player_name} might do next in natural sentences.
- Keep responses conversational and flowing, not structured like a menu.

# Safety Rules
- No violence beyond light fantasy combat
- No sexual content, horror, or harsh negativity
- Always allow recovery; no hard failure states

REMEMBER: If previous messages exist above, you are mid-conversation. Continue naturally. Do NOT re-introduce yourself or restart the game."""

app = FastAPI(title="Nanny RPG Server")
logger = logging.getLogger("nanny")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # exc.errors() may include `input` as bytes when the body fails to parse
    # (e.g. wrong Content-Type). Sanitize before JSON-encoding so we don't 500
    # the error handler itself.
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


# Global state
generator = None   # TritonGenerator instance (uses triton_kernels for MoE/matmul)
encoding = None    # Harmony encoding for chat formatting
sessions: dict[str, list[dict[str, str]]] = {}  # user_id -> full message list

# Long-term memory: player_status history per user_id, pushed from game server as array
# Stored as-is (array, ordered by id DESC from DB)
player_status_history: dict[str, list[dict]] = {}  # user_id -> [row, row, ...]
nicknames: dict[str, str] = {}  # user_id -> nickname
user_langs: dict[str, str] = {}  # user_id -> lang code (e.g. "ja", "en-US")
unlocked_zones: dict[str, list[int]] = {}  # user_id -> [zone_id, ...]
scenario_progress: dict[str, dict] = {}  # user_id -> {zone_id: {current, completed, current_step}}
user_scenarios: dict[str, str] = {}  # user_id -> scenario_id (top-level assistant persona)


# =====================================================================
# Tools  (Harmony function calling — exposed to the model via developer msg)
# =====================================================================

_EMPTY_PARAMS = {"type": "object", "properties": {}}

TOOL_DESCRIPTIONS = [
    ToolDescription.new(
        "get_game_state",
        "Get the current player's live game state and recent status history "
        "(HP, MP, level, gold, location). No arguments — operates on the current chat user.",
        parameters=_EMPTY_PARAMS,
    ),
    ToolDescription.new(
        "get_scenario_progress",
        "Get the current player's per-zone scenario progress (current step, completed flag). "
        "No arguments — operates on the current chat user.",
        parameters=_EMPTY_PARAMS,
    ),
    ToolDescription.new(
        "reset_session",
        "Wipe the current player's conversation session history. "
        "No arguments. Use only when the player explicitly asks to start over.",
        parameters=_EMPTY_PARAMS,
    ),
]


def run_function_tool(name: str, args: dict, *, user_id: str) -> dict:
    """Dispatch a Harmony function tool call.
    `user_id` is taken from the /chat request context, never from the model.
    `args` is accepted but ignored (kept for forward compatibility)."""
    if name == "get_game_state":
        return {"user_id": user_id, "rows": player_status_history.get(user_id, [])}
    if name == "get_scenario_progress":
        return {"user_id": user_id, "progress": scenario_progress.get(user_id, {})}
    if name == "reset_session":
        sessions.pop(user_id, None)
        path = _session_path(user_id)
        if path.exists():
            path.unlink()
        return {"status": "reset", "user_id": user_id}
    return {"error": f"unknown tool: {name}"}


def _get_current_scenario_voice(user_id: str) -> str:
    """Return the voice/gender from the active zone-script step.
    Falls back to the user's top-level scenario default_voice, then 'female'."""
    # Compute the top-level scenario default first so it acts as the fallback.
    sid = user_scenarios.get(user_id) or DEFAULT_SCENARIO_ID
    scenario = _load_scenario_or_fallback(sid)
    fallback_voice = (scenario or {}).get("default_voice") or "female"

    progress = scenario_progress.get(user_id)
    if not progress:
        return fallback_voice
    # Find the first non-completed zone's current_step
    for _zone_id, zone_info in progress.items():
        if zone_info.get("completed"):
            continue
        step = zone_info.get("current_step")
        if step:
            return step.get("voice", fallback_voice)
    return fallback_voice


# EXP table (cumulative EXP required for each level)
EXP_TABLE = [
    0, 0, 7, 23, 47, 110, 220, 450, 800, 1300,
    2000, 2900, 4000, 5500, 7500, 10000, 13000, 17000, 22000, 29000,
    38000, 48000, 60000, 75000, 90000, 105000, 120000, 135000, 150000, 165000,
    180000,
]

# Game config state — populated by POST /game_config from game server at startup.
# Persisted as JSON files under nanny_data/environments/.
# All reads go through _load_game_config(); writes through _save_game_config().
ENVIRONMENTS_DIR = DATA_DIR / "environments"

# File paths for each config component
_ENV_ZONES_FILE = ENVIRONMENTS_DIR / "zones.json"
_ENV_ZONE_NAMES_FILE = ENVIRONMENTS_DIR / "zone_names.json"
_ENV_ZONE_UNLOCK_FILE = ENVIRONMENTS_DIR / "zone_unlock_requirements.json"
_ENV_ENEMY_STATS_FILE = ENVIRONMENTS_DIR / "enemy_stats.json"

# Zone descriptions for system prompt (static flavor text, keyed by zone_id)
ZONE_DESCRIPTIONS: dict[int, str] = {}


def _flock_write_json(path: Path, data) -> None:
    """Write JSON to file with exclusive flock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _flock_read_json(path: Path, default=None):
    """Read JSON from file with shared flock. Returns default if file missing."""
    if not path.exists():
        return default
    with open(path, "r") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            return json.load(f)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _game_config_ready() -> bool:
    """Check if game config has been set up (files exist)."""
    return _ENV_ZONES_FILE.exists()


def _load_zone_names() -> dict[int, str]:
    raw = _flock_read_json(_ENV_ZONE_NAMES_FILE, {})
    return {int(k): v for k, v in raw.items()}


def _load_zone_unlock_requirements() -> dict[int, dict]:
    raw = _flock_read_json(_ENV_ZONE_UNLOCK_FILE, {})
    return {int(k): v for k, v in raw.items()}


def _load_enemy_stats() -> dict[str, dict]:
    return _flock_read_json(_ENV_ENEMY_STATS_FILE, {})


def _scenario_path(scenario_id: str) -> Path:
    return SCENARIOS_DIR / f"{scenario_id}.json"


def _save_scenario(scenario: dict) -> None:
    """Persist a single scenario definition to nanny_data/scenarios/<id>.json."""
    sid = scenario.get("scenario_id")
    if not sid:
        raise ValueError("scenario must have scenario_id")
    _flock_write_json(_scenario_path(sid), scenario)


def _load_scenario(scenario_id: str) -> Optional[dict]:
    """Load a scenario by id from disk; returns None if not found."""
    return _flock_read_json(_scenario_path(scenario_id), None)


def _load_scenario_or_fallback(scenario_id: Optional[str]) -> Optional[dict]:
    """Load requested scenario; if missing, fall back to DEFAULT_SCENARIO_ID,
    then to the first scenario file found. Returns None if no scenarios exist."""
    if scenario_id:
        sc = _load_scenario(scenario_id)
        if sc:
            return sc
        logger.warning("scenario %s not found; falling back to default", scenario_id)
    sc = _load_scenario(DEFAULT_SCENARIO_ID)
    if sc:
        return sc
    if SCENARIOS_DIR.exists():
        for p in sorted(SCENARIOS_DIR.glob("*.json")):
            return _flock_read_json(p, None)
    return None


def _format_zone_enemies(zone_id: int, player_attack: int = 3, player_defense: int = 1) -> str:
    """Format zone enemy table from scenario spawn actions for the system prompt."""
    zones = _flock_read_json(_ENV_ZONES_FILE, [])
    enemy_stats = _load_enemy_stats()

    # Find the zone's scenarios
    zone_scenarios = None
    for z in zones:
        if z.get("zone_id") == zone_id:
            zone_scenarios = z.get("scenarios")
            break
    if not zone_scenarios:
        return ""

    # Aggregate spawn actions: {enemy_type: total_count}
    spawn_totals: dict[str, int] = {}
    for step in zone_scenarios:
        if step.get("action") == "spawn":
            enemy_type = step.get("enemy", "")
            count = step.get("count", 1)
            if enemy_type:
                spawn_totals[enemy_type] = spawn_totals.get(enemy_type, 0) + count

    if not spawn_totals:
        return ""

    lines = [
        "\n## Enemies in this zone (from scenario spawns)",
        "Use these stats to judge whether the player should fight, flee, or be cautious.",
        "Damage formula: max(1, attacker_ATK - defender_DEF).",
    ]
    for enemy_type, total_count in spawn_totals.items():
        s = enemy_stats.get(enemy_type)
        if not s:
            lines.append(f"\n### {enemy_type} (x{total_count})")
            continue
        enemy_dmg_to_player = max(1, s["attack"] - player_defense)
        player_dmg_to_enemy = max(1, player_attack - s["defense"])
        hits_to_kill = (s["hp"] + player_dmg_to_enemy - 1) // player_dmg_to_enemy
        lines.append(f"\n### {enemy_type} (x{total_count} total in scenario)")
        lines.append(f"  HP: {s['hp']} | ATK: {s['attack']} | DEF: {s['defense']}")
        lines.append(f"  Speed: {s['speed']} | Attack range: {s['attackRange']} | Aggro range: {s['aggroRange']}")
        lines.append(f"  Critical rate: {s['criticalRate']*100:.0f}%")
        lines.append(f"  Rewards: {s['exp']} EXP, {s['goldDrop']} Gold")
        lines.append(f"  -- vs player now: enemy deals {enemy_dmg_to_player} dmg/hit, player deals {player_dmg_to_enemy} dmg/hit, ~{hits_to_kill} hits to kill")
    return "\n".join(lines)


def _format_exp_table(current_lv: int) -> str:
    """Format nearby EXP milestones for the system prompt."""
    lines = ["\n## EXP Table (cumulative EXP needed to reach each level)"]
    start = max(1, current_lv - 1)
    end = min(len(EXP_TABLE), current_lv + 4)
    for lv in range(start, end):
        marker = " <-- current" if lv == current_lv else ""
        lines.append(f"  Lv{lv}: {EXP_TABLE[lv]} EXP{marker}")
    return "\n".join(lines)


def _format_scenario_context(user_id: str, zone_id: int) -> str:
    """Format scenario context for the current zone, showing where the player is in the script."""
    # Load scenarios from zones.json
    zones = _flock_read_json(_ENV_ZONES_FILE, [])
    zone_scenarios = None
    zone_name = ""
    for z in zones:
        if z.get("zone_id") == zone_id:
            zone_scenarios = z.get("scenarios")
            zone_name = z.get("name", f"Zone {zone_id}")
            break
    if not zone_scenarios:
        return ""

    progress = scenario_progress.get(user_id, {})
    zone_prog = progress.get(str(zone_id), {})
    current_step_id = zone_prog.get("current")
    completed = zone_prog.get("completed", False)

    lines = [f"\n# Scenario — {zone_name}"]
    if completed:
        lines.append("Status: COMPLETED (all steps done)")
        return "\n".join(lines)

    # Find current step index
    current_idx = 0
    if current_step_id:
        for i, s in enumerate(zone_scenarios):
            if s.get("step_id") == current_step_id:
                current_idx = i
                break

    # Show a window: 2 completed + current + 2 upcoming
    start = max(0, current_idx - 2)
    end = min(len(zone_scenarios), current_idx + 3)
    lines.append(f"Progress: step {current_step_id or '000'} of {len(zone_scenarios)} steps")

    for i in range(start, end):
        s = zone_scenarios[i]
        sid = s.get("step_id", str(i).zfill(3))
        action = s.get("action", "?")
        msg = s.get("message", "")
        voice = s.get("voice", "")
        marker = " <<< CURRENT" if i == current_idx else ""
        if action == "message" and msg:
            lines.append(f"  [{sid}] {action}: \"{msg}\" (voice={voice}){marker}")
        else:
            lines.append(f"  [{sid}] {action}{marker}")

    lines.append("IMPORTANT: Follow this scenario. Do NOT narrate anything beyond the current step.")
    return "\n".join(lines)


def _format_zone_progress(user_id: str, unlocked: list[int]) -> str:
    """Format zone unlock progress and next goal for the system prompt.

    Unlock eligibility is decided by the game server (scenario completion +
    POI visits). Here we just reflect the resulting state and show scenario
    step progress per unlocked zone, plus a Next Goal pointing at the
    latest unlocked zone's scenario completion.
    """
    zone_names = _load_zone_names()
    zones = _flock_read_json(_ENV_ZONES_FILE, [])
    zone_by_id = {z.get("zone_id"): z for z in zones if z.get("zone_id") is not None}
    progress = scenario_progress.get(user_id, {}) or {}
    unlocked_set = set(unlocked)

    def _step_position(zid: int) -> tuple[int, int, bool]:
        """Return (current_idx_1based, total, completed) for a zone's scenario."""
        steps = (zone_by_id.get(zid, {}) or {}).get("scenarios") or []
        zp = progress.get(str(zid), {}) or {}
        is_completed = bool(zp.get("completed"))
        total = len(steps)
        current_step_id = zp.get("current")
        idx = 0
        if current_step_id and steps:
            for i, s in enumerate(steps):
                if s.get("id") == current_step_id or s.get("step_id") == current_step_id:
                    idx = i
                    break
        return (idx + 1 if total else 0, total, is_completed)

    lines = ["\n## Zone Progress"]
    next_locked: Optional[tuple[int, str]] = None

    for zid in sorted(zone_names.keys()):
        name = zone_names.get(zid, f"Zone {zid}")
        if zid in unlocked_set:
            cur_pos, total, is_completed = _step_position(zid)
            if is_completed:
                lines.append(f"  {name} (zone {zid}): UNLOCKED — scenario complete")
            elif total:
                lines.append(f"  {name} (zone {zid}): UNLOCKED — scenario step {cur_pos}/{total}")
            else:
                lines.append(f"  {name} (zone {zid}): UNLOCKED")
        else:
            lines.append(f"  {name} (zone {zid}): LOCKED — unlocks after completing the previous zone's scenario")
            if next_locked is None:
                next_locked = (zid, name)

    if next_locked is not None and unlocked_set:
        focus_zid = max(unlocked_set)
        focus_name = zone_names.get(focus_zid, f"Zone {focus_zid}")
        cur_pos, total, is_completed = _step_position(focus_zid)
        _, next_name = next_locked
        if is_completed:
            lines.append(f"\n## Next Goal\n{focus_name} scenario is complete — entering {next_name} should unlock it.")
        elif total:
            remaining = max(0, total - cur_pos)
            lines.append(
                f"\n## Next Goal\nComplete {focus_name}'s scenario "
                f"({remaining} steps remaining, currently at {cur_pos}/{total}) to unlock {next_name}."
            )
        else:
            lines.append(f"\n## Next Goal\nComplete {focus_name}'s scenario to unlock {next_name}.")
        lines.append("Encourage the player toward finishing the current scenario.")

    return "\n".join(lines)


# --- Pydantic models ---

class InlinePlayerStatus(BaseModel):
    """Player status sent inline with each /chat request from game server."""
    lv: int
    hp: int
    maxHp: int
    mp: int
    maxMp: int
    attack: int
    defense: int
    exp: int
    gold: int


class ChatRequest(BaseModel):
    user_id: str
    message: str
    nickname: Optional[str] = None
    lang: Optional[str] = None  # e.g. "ja", "en-US"
    playerStatus: Optional[InlinePlayerStatus] = None
    unlocked_zones: Optional[list[int]] = None
    unlockedZones: Optional[list[int]] = None  # alias: JS sends camelCase
    scenario_id: Optional[str] = None  # top-level assistant persona; falls back to DEFAULT_SCENARIO_ID
    scenarioId: Optional[str] = None   # alias: JS sends camelCase


class ChatResponse(BaseModel):
    user_id: str
    reply: str
    gender: str = "female"   # TTS voice: "male" or "female"
    action: str = "message"  # scenario action type


class ZoneRow(BaseModel):
    """Row from MySQL zones table."""
    zone_id: int
    name: str
    wall_height: Optional[float] = None
    unlock_condition: Optional[str] = None   # JSON string or null
    permission: Optional[str] = None         # JSON string e.g. '{"lv": 4, "gold": 100}'
    scenarios: Optional[list] = None         # scenario steps for this zone


class ScenarioRow(BaseModel):
    """Top-level scenario / assistant persona pushed by the game server.

    Each scenario defines who the LLM IS for that learning track
    (e.g. mobile-core-network senior engineer). The persona_prompt is
    injected into the system prompt; common RPG mechanics stay in code.

    response_rules controls how the assistant formats replies (length cap,
    ending tone, etc.). If omitted, a sensible default is used.
    """
    scenario_id: str
    name: str
    persona_prompt: str
    response_rules: Optional[str] = None  # bullet-list text injected into "Response Format Rules"
    default_voice: Optional[str] = "female"  # TTS voice fallback
    description: Optional[str] = None


class GameConfigRequest(BaseModel):
    """Game server pushes zones + scenarios at startup."""
    zones: list[ZoneRow]
    scenarios: Optional[list[ScenarioRow]] = None


class PlayerStatusRow(BaseModel):
    """Single row from MySQL player_status table."""
    id: int
    user_id: str
    zone_id: int
    lv: int
    hp: int
    max_hp: int
    mp: int
    max_mp: int
    attack: int
    defense: int
    exp: int
    gold: int
    pos_x: float
    pos_z: float
    is_dead: int
    updated_at: str


class GameEvent(BaseModel):
    """Game event notification from the game server."""
    user_id: str
    event_type: str          # e.g. "combat", "level_up", "item_pickup", "quest", "zone_change", "death", "respawn"
    description: str         # human-readable description
    player_status: list[PlayerStatusRow]  # current status array from DB
    scenario_progress: Optional[dict] = None  # {zone_id: {current, completed, current_step:{id,action,text,voice,tts}}}


# =====================================================================
# Long-term memory  (game server → POST /game_state)
# =====================================================================

def status_to_prompt_block(rows: list[dict]) -> str:
    """
    Format player_status array into a prompt block.
    rows[0] = latest (highest id), rest = recent history for context.
    """
    if not rows:
        return ""

    zone_names = _load_zone_names()
    latest = rows[0]
    zone = zone_names.get(latest.get("zone_id", 0), f"Zone {latest.get('zone_id', '?')}")
    dead_str = " ** DEAD — needs to be revived! **" if latest.get("is_dead") else ""

    block = f"""
# Live Game State (from game server DB — this is the authoritative source of truth)
## Current (id={latest['id']})
Level: {latest['lv']} | HP: {latest['hp']}/{latest['max_hp']} | MP: {latest['mp']}/{latest['max_mp']}
Attack: {latest['attack']} | Defense: {latest['defense']}
EXP: {latest['exp']} | Gold: {latest['gold']}
Location: {zone} (x={latest['pos_x']:.1f}, z={latest['pos_z']:.1f})
Status: {"Dead" if latest.get('is_dead') else "Alive"}{dead_str}
Last updated: {latest['updated_at']}
"""

    # Show recent changes so the nanny understands what happened
    if len(rows) > 1:
        block += "\n## Recent Status History (newest first)\n"
        for row in rows[1:10]:  # show up to 9 previous entries
            z = zone_names.get(row.get("zone_id", 0), f"Zone {row.get('zone_id', '?')}")
            block += (
                f"  id={row['id']}: Lv{row['lv']} HP{row['hp']}/{row['max_hp']} "
                f"MP{row['mp']}/{row['max_mp']} ATK{row['attack']} DEF{row['defense']} "
                f"Gold{row['gold']} EXP{row['exp']} {z} "
                f"{'DEAD' if row.get('is_dead') else 'alive'} "
                f"({row['updated_at']})\n"
            )

    return block


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
# Prompt construction  (system + live game state + short-term window)
# =====================================================================

def build_prompt_messages(
    user_id: str,
    history: list[dict[str, str]],
    inline_status: Optional[dict] = None,
) -> list[dict[str, str]]:
    """
    Build the message list sent to the model:
      1. System prompt + live game state from game server (if available)
      2. If history > SHORT_TERM_WINDOW, a note about skipped messages
      3. Last SHORT_TERM_WINDOW messages (short-term memory)
    """
    player_name = nicknames.get(user_id, "Adventurer")
    user_lang = user_langs.get(user_id, "en-US")

    # Build language instruction based on user's lang setting
    lang_code = user_lang.split("-")[0].lower()  # "ja", "en", etc.
    if lang_code == "ja":
        lang_instruction = (
            "All in-game narration, dialogue, and system messages MUST be in Japanese (日本語). "
            "The player ({player_name}) inputs in Japanese. "
            "Respond entirely in Japanese. "
            "NEVER use emojis or emoticons."
        ).replace("{player_name}", player_name)
    elif lang_code == "en":
        lang_instruction = (
            "All in-game narration, dialogue, and system messages MUST be in English. "
            "The player ({player_name}) inputs in English. "
            "NEVER use emojis, emoticons, or any non-ASCII characters. Use only plain ASCII text."
        ).replace("{player_name}", player_name)
    else:
        lang_instruction = (
            f"All in-game narration, dialogue, and system messages MUST be in the language with code '{user_lang}'. "
            f"The player ({player_name}) inputs in that language. "
            f"Respond entirely in that language. "
            f"NEVER use emojis, emoticons, or any non-ASCII characters. Use only plain ASCII text."
        ).replace("{player_name}", player_name)

    # Resolve scenario persona (top-level assistant identity)
    sid = user_scenarios.get(user_id) or DEFAULT_SCENARIO_ID
    scenario = _load_scenario_or_fallback(sid)
    default_response_rules = (
        "- MAXIMUM 10 words per response. Be extremely brief.\n"
        "- Do NOT end your response with a question. Just narrate what happens."
    )
    if scenario:
        scenario_name = scenario.get("name", scenario.get("scenario_id", sid))
        scenario_persona = scenario.get("persona_prompt", "")
        scenario_response_rules = scenario.get("response_rules") or default_response_rules
    else:
        # No scenarios pushed yet — use a minimal placeholder so prompt still renders.
        # Game server should POST /game_config with scenarios at startup.
        scenario_name = sid
        scenario_persona = "You are the player's assistant in this RPG."
        scenario_response_rules = default_response_rules
        logger.warning("no scenario found for user=%s sid=%s; using placeholder persona", user_id, sid)

    # Substitute scenario fields first so any {player_name} / {lang_instruction}
    # references inside persona_prompt or response_rules are resolved by the later replacements.
    system_content = (
        SYSTEM_PROMPT
        .replace("{scenario_name}", scenario_name)
        .replace("{scenario_persona}", scenario_persona)
        .replace("{scenario_response_rules}", scenario_response_rules)
        .replace("{player_name}", player_name)
        .replace("{lang_instruction}", lang_instruction)
    )

    # Determine current player stats from available sources
    current_lv = 1
    zone_id = 1
    player_attack = 3   # default starting attack
    player_defense = 1  # default starting defense

    # Inject live game state from game server DB (POST /game_state history)
    rows = player_status_history.get(user_id)
    if rows:
        system_content += "\n" + status_to_prompt_block(rows)
        zone_id = rows[0].get("zone_id", 1)
        current_lv = rows[0].get("lv", 1)
        player_attack = rows[0].get("attack", player_attack)
        player_defense = rows[0].get("defense", player_defense)

    # Inject inline player status from /chat request (real-time snapshot)
    if inline_status:
        current_lv = inline_status.get("lv", current_lv)
        player_attack = inline_status.get("attack", player_attack)
        player_defense = inline_status.get("defense", player_defense)
        system_content += f"""
# Inline Player Status (sent with this request)
Level: {inline_status['lv']} | HP: {inline_status['hp']}/{inline_status['maxHp']} | MP: {inline_status['mp']}/{inline_status['maxMp']}
Attack: {inline_status['attack']} | Defense: {inline_status['defense']}
EXP: {inline_status['exp']} | Gold: {inline_status['gold']}
"""

    zone_name = _load_zone_names().get(zone_id, f"Zone {zone_id}")
    zone_desc = ZONE_DESCRIPTIONS.get(zone_id, f"## Current Zone: {zone_name}")
    system_content += "\n" + zone_desc
    system_content += _format_zone_enemies(zone_id, player_attack, player_defense)
    system_content += _format_exp_table(current_lv)

    # Inject zone progress and next goal
    user_unlocked = unlocked_zones.get(user_id, [1])  # default: only zone 1
    system_content += _format_zone_progress(user_id, user_unlocked)
    system_content += _format_scenario_context(user_id, zone_id)

    messages = [{"role": "system", "content": system_content}]

    if len(history) > SHORT_TERM_WINDOW:
        skipped = len(history) - SHORT_TERM_WINDOW
        messages.append({
            "role": "system",
            "content": (
                f"[{skipped} earlier messages omitted. "
                f"Refer to the Live Game State above for current stats.]"
            ),
        })
        recent = history[-SHORT_TERM_WINDOW:]
    else:
        recent = history

    # Sanitize turn structure: merge consecutive same-role messages,
    # condense repetitive game events, and drop empty turns.
    sanitized: list[dict[str, str]] = []
    for msg in recent:
        content = msg.get("content", "").strip()
        if not content:
            continue
        if sanitized and sanitized[-1]["role"] == msg["role"]:
            # Merge consecutive messages of the same role
            sanitized[-1] = {
                "role": msg["role"],
                "content": sanitized[-1]["content"] + "\n" + content,
            }
        else:
            sanitized.append({"role": msg["role"], "content": content})

    # Condense game event blocks: count repeated enemy_killed lines
    for i, msg in enumerate(sanitized):
        if msg["role"] != "system" or "[Game Event:" not in msg["content"]:
            continue
        lines = msg["content"].split("\n")
        if len(lines) <= 3:
            continue
        # Count kills by enemy type
        kill_counts: dict[str, int] = {}
        other_lines: list[str] = []
        for line in lines:
            if "[Game Event: enemy_killed] Defeated " in line:
                enemy = line.split("Defeated ", 1)[1].strip()
                kill_counts[enemy] = kill_counts.get(enemy, 0) + 1
            else:
                other_lines.append(line)
        if kill_counts:
            summary_parts = [f"{name} x{count}" for name, count in kill_counts.items()]
            other_lines.append(f"[Game Event: enemy_killed] Defeated: {', '.join(summary_parts)}")
        sanitized[i] = {"role": msg["role"], "content": "\n".join(other_lines)}

    messages.extend(sanitized)
    logger.info(
        "prompt: user=%s, history=%d msgs, sending=%d msgs (system+%d recent, sanitized from %d)",
        user_id, len(history), len(messages), len(sanitized), len(recent),
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
      {"role": "assistant", "content": str, "channel"?: str, "recipient"?: str}
      {"role": "tool",      "name": str,    "content": str}   # tool result
    """
    harmony_msgs: list[Message] = []
    first_system = True

    for msg in prompt_messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            if first_system:
                first_system = False
                # Model-level system message (identity, reasoning, channel policy)
                sys_content = (
                    SystemContent.new()
                    .with_reasoning_effort(ReasoningEffort.LOW)
                    .with_conversation_start_date(
                        datetime.datetime.now().strftime("%Y-%m-%d")
                    )
                    .with_required_channels(["analysis", "commentary", "final"])
                )
                harmony_msgs.append(
                    Message.from_role_and_content(Role.SYSTEM, sys_content)
                )
                # Custom instructions + tools as developer message
                dev_content = (
                    DeveloperContent.new()
                    .with_instructions(content)
                    .with_function_tools(TOOL_DESCRIPTIONS)
                )
                harmony_msgs.append(
                    Message.from_role_and_content(Role.DEVELOPER, dev_content)
                )
            else:
                # Additional system messages (game events, etc.)
                dev_content = DeveloperContent.new().with_instructions(content)
                harmony_msgs.append(
                    Message.from_role_and_content(Role.DEVELOPER, dev_content)
                )
        elif role == "user":
            harmony_msgs.append(Message.from_role_and_content(Role.USER, content))
        elif role == "assistant":
            m = Message.from_role_and_content(Role.ASSISTANT, content)
            ch = msg.get("channel", "final")
            m = m.with_channel(ch)
            rcpt = msg.get("recipient")
            if rcpt:
                m = m.with_recipient(rcpt)
            harmony_msgs.append(m)
        elif role == "tool":
            tool_name = msg.get("name", "unknown")
            m = (
                Message.from_author_and_content(
                    Author.new(Role.TOOL, f"functions.{tool_name}"),
                    content,
                )
                .with_channel("commentary")
                .with_recipient("assistant")
            )
            harmony_msgs.append(m)

    return harmony_msgs


def generate_once(
    prompt_messages: list[dict],
    max_new_tokens: int = 256,
    max_reply_tokens: int = 80,
) -> dict:
    """
    Run one forward pass and return either a final reply or a tool call.

    Returns one of:
      {"type": "final",     "reply": str, "truncated": bool}
      {"type": "tool_call", "recipient": str, "arguments": str}

    `recipient` is the full Harmony name like "functions.get_game_state".
    """
    t0 = time.perf_counter()

    harmony_msgs = _to_harmony_messages(prompt_messages)
    conversation = Conversation.from_messages(harmony_msgs)
    tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    stop_tokens = encoding.stop_tokens_for_assistant_actions()

    decoded_prompt = encoding.decode(tokens)
    logger.info("=== PROMPT DUMP (last 500 chars) ===\n%s", decoded_prompt[-500:])

    t1 = time.perf_counter()

    parser = StreamableParser(encoding, role=Role.ASSISTANT)
    final_parts: list[str] = []
    tool_parts: list[str] = []
    last_tool_recipient: Optional[str] = None
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
        elif ch == "commentary":
            rcpt = parser.current_recipient
            if rcpt and rcpt.startswith("functions."):
                tool_parts.append(delta)
                last_tool_recipient = rcpt

    t2 = time.perf_counter()

    gen_ms = (t2 - t1) * 1000
    tok_per_sec = gen_count / (t2 - t1) if (t2 - t1) > 0 else 0

    if last_tool_recipient and not final_parts:
        args_str = "".join(tool_parts)
        logger.info(
            "generate_once: tool_call recipient=%s args_len=%d "
            "input=%d output=%d encode=%.1fms gen=%.1fms (%.1f tok/s) channels=%s",
            last_tool_recipient, len(args_str),
            len(tokens), gen_count, (t1 - t0) * 1000, gen_ms, tok_per_sec, channels_seen,
        )
        return {
            "type": "tool_call",
            "recipient": last_tool_recipient,
            "arguments": args_str,
        }

    reply = "".join(final_parts)
    logger.info(
        "generate_once: final reply_len=%d truncated=%s "
        "input=%d output=%d (final=%d) encode=%.1fms gen=%.1fms (%.1f tok/s) channels=%s",
        len(reply), truncated,
        len(tokens), gen_count, final_token_count,
        (t1 - t0) * 1000, gen_ms, tok_per_sec, channels_seen,
    )
    return {"type": "final", "reply": reply, "truncated": truncated}


def agent_generate_reply(
    prompt_messages: list[dict],
    *,
    user_id: str,
    max_steps: int = 4,
) -> tuple[str, bool]:
    """
    Run the analysis → tool_call → tool_result → final loop.
    Returns (reply, was_truncated) matching the old generate_reply signature.

    Tool-call exchanges live only inside this call's local message list and
    are NOT persisted to the session history (they are regenerable on demand).
    """
    msgs = list(prompt_messages)

    for step in range(max_steps):
        out = generate_once(msgs)
        if out["type"] == "final":
            return out["reply"], out["truncated"]

        recipient = out["recipient"]                      # "functions.<name>"
        raw_args = out["arguments"]
        tool_name = recipient.split(".", 1)[1] if "." in recipient else recipient

        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            result = {"error": "invalid JSON arguments", "raw": raw_args[:200]}
        else:
            try:
                result = run_function_tool(tool_name, args, user_id=user_id)
            except Exception as e:
                logger.exception("tool %s raised", tool_name)
                result = {"error": f"{type(e).__name__}: {e}"}

        logger.info(
            "agent_step=%d tool=%s args=%s result_keys=%s",
            step, tool_name, args if isinstance(args, dict) else raw_args[:120],
            list(result.keys()) if isinstance(result, dict) else type(result).__name__,
        )

        # Echo the assistant's tool call so the next pass sees a coherent transcript.
        msgs.append({
            "role": "assistant",
            "channel": "commentary",
            "recipient": recipient,
            "content": raw_args,
        })
        msgs.append({
            "role": "tool",
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        })

    logger.warning("agent loop exhausted after %d steps without final", max_steps)
    return "", False


# =====================================================================
# Endpoints
# =====================================================================

@app.post("/game_config")
async def receive_game_config(cfg: GameConfigRequest):
    """
    Game server pushes zones + scenarios at startup.
    Persists to nanny_data/environments/ and nanny_data/scenarios/ as JSON
    files with flock. All consumers read from these files on each request.
    """
    # --- Build derived dicts from zones rows ---
    new_zone_names: dict[int, str] = {}
    new_zone_unlock: dict[int, dict] = {}
    raw_zones: list[dict] = []

    for z in cfg.zones:
        new_zone_names[z.zone_id] = z.name
        raw_zones.append(z.model_dump())

        # Parse permission JSON -> unlock requirements
        if z.permission:
            try:
                perm = json.loads(z.permission) if isinstance(z.permission, str) else z.permission
            except (json.JSONDecodeError, TypeError):
                perm = {}
            min_lv = perm.get("lv", 0)
            min_gold = perm.get("gold", 0)
            if min_lv > 0 or min_gold > 0:
                new_zone_unlock[z.zone_id] = {
                    "min_lv": min_lv,
                    "min_gold": min_gold,
                    "description": f"Reach Level {min_lv} and {min_gold} Gold to enter {z.name}",
                }

        # Parse unlock_condition JSON if present
        if z.unlock_condition:
            try:
                cond = json.loads(z.unlock_condition) if isinstance(z.unlock_condition, str) else z.unlock_condition
            except (json.JSONDecodeError, TypeError):
                cond = None
            if cond and z.zone_id in new_zone_unlock:
                new_zone_unlock[z.zone_id]["unlock_condition"] = cond
            elif cond:
                new_zone_unlock[z.zone_id] = {"min_lv": 0, "min_gold": 0, "unlock_condition": cond}

    # --- Persist all to JSON files with flock ---
    _flock_write_json(_ENV_ZONES_FILE, raw_zones)
    _flock_write_json(_ENV_ZONE_NAMES_FILE, new_zone_names)
    _flock_write_json(_ENV_ZONE_UNLOCK_FILE, new_zone_unlock)

    # --- Persist scenarios (one file per scenario_id) ---
    scenario_ids: list[str] = []
    if cfg.scenarios:
        SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
        for sc in cfg.scenarios:
            _save_scenario(sc.model_dump())
            scenario_ids.append(sc.scenario_id)

    logger.info(
        "game_config: loaded %d zones, %d scenarios. ZONE_NAMES=%s SCENARIOS=%s",
        len(cfg.zones), len(scenario_ids), new_zone_names, scenario_ids,
    )
    return {
        "status": "ok",
        "zones": len(cfg.zones),
        "scenarios": scenario_ids,
        "zone_names": new_zone_names,
    }


@app.post("/game_state")
async def receive_game_state(rows: list[PlayerStatusRow]):
    """
    Game server pushes player_status array (from DB, ORDER BY id DESC LIMIT 100).
    Stored as-is per user_id. This is the authoritative long-term memory.
    """
    if not rows:
        return {"status": "empty"}
    user_id = rows[0].user_id
    player_status_history[user_id] = [r.model_dump() for r in rows]
    return {"status": "ok", "user_id": user_id, "rows": len(rows)}


@app.post("/game_event")
async def receive_game_event(event: GameEvent):
    """
    Game server pushes event + player_status array.
    The event description is injected into the conversation as a system message
    so the nanny can react naturally to what just happened in-game.
    """
    if not _game_config_ready():
        return JSONResponse(
            status_code=503,
            content={"detail": "Service not ready: game_config has not been set up. POST /game_config first."},
        )
    # Update player status history
    if event.player_status:
        user_id = event.user_id
        player_status_history[user_id] = [r.model_dump() for r in event.player_status]

    # Update scenario progress
    if event.scenario_progress:
        scenario_progress[event.user_id] = event.scenario_progress
        logger.info("scenario_progress updated for user=%s: %s", event.user_id, event.scenario_progress)

    # Inject event into conversation history as a system-level note.
    # For high-frequency events (enemy_killed), merge consecutive ones into
    # a single summary to avoid flooding the history window.
    history = get_or_create_session(event.user_id)

    event_msg = f"[Game Event: {event.event_type}] {event.description}"

    if event.event_type == "enemy_killed":
        # Merge with the last message if it's also an enemy_killed event
        if (history
                and history[-1]["role"] == "system"
                and history[-1]["content"].startswith("[Game Event: enemy_killed]")):
            # Append to existing kill summary
            history[-1]["content"] += f"\n{event_msg}"
        else:
            history.append({"role": "system", "content": event_msg})
    else:
        history.append({"role": "system", "content": event_msg})

    save_session(event.user_id, history)

    return {"status": "ok", "user_id": event.user_id, "event_type": event.event_type}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not _game_config_ready():
        return JSONResponse(
            status_code=503,
            content={"detail": "Service not ready: game_config has not been set up. POST /game_config first."},
        )
    logger.info("chat request: %s", req.model_dump())
    # Store nickname if provided
    if req.nickname:
        nicknames[req.user_id] = req.nickname

    # Store user language if provided
    if req.lang:
        user_langs[req.user_id] = req.lang

    # Store unlocked zones if provided (accept both snake_case and camelCase)
    uzones = req.unlocked_zones if req.unlocked_zones is not None else req.unlockedZones
    if uzones is not None:
        unlocked_zones[req.user_id] = uzones

    # Store scenario selection if provided (accept both snake_case and camelCase)
    sid = req.scenario_id if req.scenario_id is not None else req.scenarioId
    if sid:
        user_scenarios[req.user_id] = sid

    history = get_or_create_session(req.user_id)

    # Detect system-injected messages (e.g. "[System: Player HP is critically low...]")
    # These come from the game client but should be treated as system instructions, not player input.
    if req.message.startswith("[System:"):
        history.append({"role": "system", "content": req.message})
    else:
        history.append({"role": "user", "content": req.message})

    # Build prompt: system + live game state + last 100 messages
    inline_status = req.playerStatus.model_dump() if req.playerStatus else None
    prompt_messages = build_prompt_messages(req.user_id, history, inline_status=inline_status)
    reply, truncated = agent_generate_reply(prompt_messages, user_id=req.user_id)

    # Sanitize reply. For English/other: strip to latin-1 (TTS constraint).
    # For Japanese: TTS handles unicode, so keep CJK and strip only control chars.
    raw_reply = reply
    req_lang_code = (req.lang or user_langs.get(req.user_id, "en-US")).split("-")[0].lower()
    if req_lang_code == "ja":
        reply = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", reply)
        reply = re.sub(r"\s+", " ", reply).strip()
    else:
        reply = reply.encode("latin-1", errors="ignore").decode("latin-1")
        reply = re.sub(r"[^\x20-\x7E\xa0-\xff]", " ", reply)
        reply = re.sub(r" {2,}", " ", reply).strip()

    # If truncated mid-generation, trim to the last sentence terminator so we
    # don't save a mid-word fragment that the model would echo on the next turn
    # (the original source of the "same nonsense reply" loop).
    storable_reply = reply
    if truncated and reply:
        if req_lang_code == "ja":
            # Japanese sanitizer keeps CJK, so look for full-width terminators.
            # ASCII terminators are also included in case of mixed text.
            terminators = "。！？.!?"
        else:
            # Non-ja was stripped to latin-1, so only ASCII terminators survive.
            terminators = ".!?"
        last_end = max((reply.rfind(c) for c in terminators), default=-1)
        if last_end >= 0:
            storable_reply = reply[: last_end + 1]
            reply = storable_reply
        else:
            # No sentence boundary at all — the whole reply is a fragment.
            # Show it to the user (so they aren't left blank) but do NOT save
            # it to history, to prevent the echo loop on the next turn.
            storable_reply = ""
            logger.warning(
                "generate: truncated reply with no sentence boundary for user=%s; "
                "not saving to history. raw=%r",
                req.user_id, raw_reply[:200],
            )

    # Only store non-empty replies to avoid polluting history with blank assistant turns
    if storable_reply:
        history.append({"role": "assistant", "content": storable_reply})
    elif not reply:
        logger.warning(
            "generate: empty reply after latin-1 strip for user=%s. "
            "Raw reply (%d chars): %r",
            req.user_id, len(raw_reply), raw_reply[:200],
        )
    save_session(req.user_id, history)

    gender = _get_current_scenario_voice(req.user_id)
    return ChatResponse(user_id=req.user_id, reply=reply, gender=gender)


@app.post("/reset")
async def reset(user_id: str):
    if user_id in sessions:
        del sessions[user_id]
    path = _session_path(user_id)
    if path.exists():
        path.unlink()
    return {"status": "reset", "user_id": user_id}


@app.get("/game_state")
async def get_game_state(user_id: str):
    """View current player_status history from game server."""
    rows = player_status_history.get(user_id)
    if rows:
        return JSONResponse(rows)
    return JSONResponse({"error": "no status received for this user"}, status_code=404)


@app.get("/history")
async def get_history(user_id: str):
    """View full session history."""
    if user_id in sessions:
        return JSONResponse(sessions[user_id])
    history = load_session(user_id)
    if history:
        return JSONResponse(history)
    return JSONResponse({"error": "session not found"}, status_code=404)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nanny RPG - Fields of Light</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e0e0e0; height: 100vh; display: flex; flex-direction: column;
  }
  header {
    text-align: center; padding: 12px;
    background: rgba(0,0,0,0.3); border-bottom: 1px solid #e94560;
  }
  header h1 { font-size: 1.4em; color: #e94560; }
  header p { font-size: 0.85em; color: #aaa; }
  #chat {
    flex: 1; overflow-y: auto; padding: 16px;
    display: flex; flex-direction: column; gap: 10px;
  }
  .msg { max-width: 80%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: #e94560; color: #fff; border-bottom-right-radius: 2px; }
  .msg.assistant { align-self: flex-start; background: #1e2a4a; border: 1px solid #2a3a5a; border-bottom-left-radius: 2px; }
  .msg.system { align-self: center; color: #888; font-style: italic; font-size: 0.85em; }
  #input-area {
    display: flex; padding: 12px; gap: 8px;
    background: rgba(0,0,0,0.3); border-top: 1px solid #333;
  }
  #input-area input {
    flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #444;
    background: #1a1a2e; color: #e0e0e0; font-size: 1em; outline: none;
  }
  #input-area input:focus { border-color: #e94560; }
  #input-area button {
    padding: 10px 20px; border-radius: 8px; border: none;
    background: #e94560; color: #fff; font-size: 1em; cursor: pointer;
  }
  #input-area button:hover { background: #c73650; }
  #input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>Fields of Light and the Starry Town</h1>
  <p>Your Nanny awaits, {player_name}.</p>
</header>
<div id="chat"></div>
<div id="input-area">
  <input id="msg" placeholder="Type your action..." autofocus />
  <button id="send" onclick="sendMsg()">Send</button>
</div>
<script>
// user_id from URL param or generate one
const params = new URLSearchParams(location.search);
const userId = params.get('user_id') || crypto.randomUUID();
const chat = document.getElementById('chat');
const msgInput = document.getElementById('msg');
const sendBtn = document.getElementById('send');

document.querySelector('header p').textContent = 'Player: ' + userId.slice(0, 8) + '...';

function addMsg(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

async function sendMsg() {
  const text = msgInput.value.trim();
  if (!text) return;
  msgInput.value = '';
  addMsg('user', text);
  sendBtn.disabled = true;

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_id: userId, message: text})
    });
    const data = await res.json();
    addMsg('assistant', data.reply);
  } catch (e) {
    addMsg('system', 'Connection error: ' + e.message);
  }
  sendBtn.disabled = false;
  msgInput.focus();
}

msgInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendMsg(); });

// Start the game automatically
fetch('/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({user_id: userId, message: 'Start the game!'})
}).then(r => r.json()).then(data => {
  addMsg('assistant', data.reply);
});
</script>
</body>
</html>
"""


def main():
    global generator, encoding

    parser = argparse.ArgumentParser(description="Nanny RPG HTTP Server")
    parser.add_argument("--checkpoint", default="openai/gpt-oss-20b", help="Model name or path")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--data-dir", default="nanny_data", help="Directory for persistent data")
    parser.add_argument("--context-length", type=int, default=16384,
                        help="Max model context length passed to the vLLM engine")
    args = parser.parse_args()

    # Set up data directories
    global DATA_DIR, SESSIONS_DIR, ENVIRONMENTS_DIR, SCENARIOS_DIR
    global _ENV_ZONES_FILE, _ENV_ZONE_NAMES_FILE, _ENV_ZONE_UNLOCK_FILE, _ENV_ENEMY_STATS_FILE
    DATA_DIR = Path(args.data_dir)
    SESSIONS_DIR = DATA_DIR / "sessions"
    ENVIRONMENTS_DIR = DATA_DIR / "environments"
    SCENARIOS_DIR = DATA_DIR / "scenarios"
    _ENV_ZONES_FILE = ENVIRONMENTS_DIR / "zones.json"
    _ENV_ZONE_NAMES_FILE = ENVIRONMENTS_DIR / "zone_names.json"
    _ENV_ZONE_UNLOCK_FILE = ENVIRONMENTS_DIR / "zone_unlock_requirements.json"
    _ENV_ENEMY_STATS_FILE = ENVIRONMENTS_DIR / "enemy_stats.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ENVIRONMENTS_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    print("Data directory ready. Waiting for game server to POST /game_config.")

    # Load model with vLLM backend
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
