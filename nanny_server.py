"""
HTTP server for chatting with the Nanny RPG Game Master using gpt-oss-20b.

Usage:
    python nanny_server.py [--checkpoint openai/gpt-oss-20b] [--port 8080]

Endpoints:
    GET  /              — Simple web UI
    POST /chat          — JSON API: {"user_id": "...", "message": "..."} -> {"reply": "..."}
    POST /game_config   — Game server pushes zones + zone_enemies tables at startup
    POST /game_state    — Game server pushes player state (from DB) on events
    POST /reset         — Reset conversation history
    GET  /game_state    — View current player state
    GET  /history       — View session history

Memory system:
    - Short-term: Last 100 messages (user+assistant turns) kept in prompt context
    - Long-term:  Game server POSTs player_state from MySQL on each event/command
    - Sessions:   Full conversation history saved to disk per user_id
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
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
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

SYSTEM_PROMPT = """\
You are a gentle, caring mother and nanny for an 11-year-old boy named {player_name}.
At the same time, you are the Game Master (GM) of a beginner-friendly tabletop RPG.

CRITICAL RULE - READ FIRST:
- You MUST continue from where the conversation left off. Read the previous messages carefully and respond to what {player_name} just said.
- Do NOT restart the game or re-introduce yourself if there are previous messages in this conversation.
- Only introduce the game world if this is the very first message with no prior history.

IMPORTANT:
- All in-game narration, dialogue, and system messages MUST be in English.
- The player ({player_name}) inputs in English.
- Your tone must always be warm, supportive, and nurturing, like a kind mother guiding her child.
- NEVER use emojis, emoticons, or any non-ASCII characters. Use only plain ASCII text.
- Do NOT use markdown formatting (no **, no ##, no bullet points). Use plain text only.

# Core Role
You must simultaneously:
1. Act as a Game Master narrating a fantasy adventure
2. Act as a loving mother/nanny supporting and encouraging {player_name}
3. Keep the game simple and beginner-friendly
4. Guide the player gently when they are unsure

# Tone Rules
- Warm, kind, encouraging
- Simple English suitable for a child
- Never harsh, never cold
- Slight emotional support is encouraged (e.g., "That's a lovely idea, {player_name}.")
- MAXIMUM 10 words per response. Be extremely brief.
- Do NOT end your response with a question. Just narrate what happens.

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
Final goal: Collect Monster Stones and give them to you (the nanny/GM).

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
## Magic - The player can say "Cure" to restore HP (costs 1 MP). If HP drops below 30%, gently remind {player_name} to say "Cure".
## Death - If HP is 0, the player is dead. Gently remind {player_name} to say "Revive" to come back to life.
## Progression - Gain EXP + Gold from monsters. Level up increases stats. Celebrate level-ups warmly.
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
- Talk naturally, like a warm conversation between a caring mother and her child.
- Do NOT present numbered lists of choices or options (no "1. ... 2. ... 3. ...").
- Instead, describe the scene and gently suggest what {player_name} might do next in natural sentences.
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
    logger.error("Validation error on %s %s: %s", request.method, request.url.path, exc.errors())
    logger.error("Request body: %s", exc.body)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


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
unlocked_zones: dict[str, list[int]] = {}  # user_id -> [zone_id, ...]
scenario_progress: dict[str, dict] = {}  # user_id -> {zone_id: {current, completed, current_step}}


def _get_current_scenario_voice(user_id: str) -> str:
    """Return the voice/gender from the active (non-completed) scenario step, or 'female' as default."""
    progress = scenario_progress.get(user_id)
    if not progress:
        return "female"
    # Find the first non-completed zone's current_step
    for _zone_id, zone_info in progress.items():
        if zone_info.get("completed"):
            continue
        step = zone_info.get("current_step")
        if step:
            return step.get("voice", "female")
    return "female"


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
_ENV_ZONE_ENEMIES_FILE = ENVIRONMENTS_DIR / "zone_enemies.json"
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
    return _ENV_ZONES_FILE.exists() and _ENV_ZONE_ENEMIES_FILE.exists()


def _load_zone_names() -> dict[int, str]:
    raw = _flock_read_json(_ENV_ZONE_NAMES_FILE, {})
    return {int(k): v for k, v in raw.items()}


def _load_zone_unlock_requirements() -> dict[int, dict]:
    raw = _flock_read_json(_ENV_ZONE_UNLOCK_FILE, {})
    return {int(k): v for k, v in raw.items()}


def _load_enemy_stats() -> dict[str, dict]:
    return _flock_read_json(_ENV_ENEMY_STATS_FILE, {})


def _load_zone_enemies() -> dict[int, list[list]]:
    """Returns {zone_id: [[enemy_type, player_lv, count], ...]}."""
    raw = _flock_read_json(_ENV_ZONE_ENEMIES_FILE, {})
    return {int(k): v for k, v in raw.items()}


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


def _format_zone_progress(unlocked_zones: list[int], current_lv: int, current_exp: int, current_gold: int = 0) -> str:
    """Format zone unlock progress and next goal for the system prompt."""
    zone_names = _load_zone_names()
    zone_unlock = _load_zone_unlock_requirements()
    unlocked_set = set(unlocked_zones)
    lines = ["\n## Zone Progress"]

    # Show unlocked zones
    for zid in sorted(unlocked_set):
        name = zone_names.get(zid, f"Zone {zid}")
        lines.append(f"  {name} (zone {zid}): UNLOCKED")

    # Show locked zones with requirements and progress
    next_goal = None
    for zid in sorted(zone_names.keys()):
        if zid in unlocked_set:
            continue
        name = zone_names.get(zid, f"Zone {zid}")
        req = zone_unlock.get(zid)
        if req:
            min_lv = req["min_lv"]
            min_gold = req.get("min_gold", 0)
            lv_met = current_lv >= min_lv
            gold_met = current_gold >= min_gold
            if lv_met and gold_met:
                lines.append(f"  {name} (zone {zid}): LOCKED (requirements met! Lv{current_lv} >= Lv{min_lv}, Gold{current_gold} >= {min_gold}) -- may need a trigger event")
            else:
                parts = []
                if not lv_met:
                    lvs_needed = min_lv - current_lv
                    if min_lv < len(EXP_TABLE):
                        exp_needed = max(0, EXP_TABLE[min_lv] - current_exp)
                    else:
                        exp_needed = 0
                    parts.append(f"Lv{min_lv} ({lvs_needed} more levels, {exp_needed} more EXP)")
                if not gold_met:
                    gold_needed = min_gold - current_gold
                    parts.append(f"{min_gold} Gold ({gold_needed} more Gold)")
                lines.append(f"  {name} (zone {zid}): LOCKED -- need {' and '.join(parts)}")
                if next_goal is None:
                    next_goal = f"Reach Lv{min_lv} and {min_gold} Gold to unlock {name}. Need: {', '.join(parts)}."
        else:
            lines.append(f"  {name} (zone {zid}): LOCKED")

    if next_goal:
        lines.append(f"\n## Next Goal\n{next_goal}")
        lines.append("Encourage the player to keep fighting monsters and gaining EXP and Gold toward this goal.")

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
    playerStatus: Optional[InlinePlayerStatus] = None
    unlocked_zones: Optional[list[int]] = None
    unlockedZones: Optional[list[int]] = None  # alias: JS sends camelCase


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


class ZoneEnemyRow(BaseModel):
    """Row from MySQL zone_enemies table."""
    id: Optional[int] = None
    zone_id: int
    enemy_type: str
    player_lv: int = 1
    count: int = 1
    spawn_points: Optional[str] = None


class GameConfigRequest(BaseModel):
    """Game server pushes zone + zone_enemies tables at startup."""
    zones: list[ZoneRow]
    zone_enemies: list[ZoneEnemyRow]


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
    system_content = SYSTEM_PROMPT.replace("{player_name}", player_name)

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
    current_exp = 0
    current_gold = 0
    if inline_status:
        current_exp = inline_status.get("exp", 0)
        current_gold = inline_status.get("gold", 0)
    elif rows:
        current_exp = rows[0].get("exp", 0)
        current_gold = rows[0].get("gold", 0)
    system_content += _format_zone_progress(user_unlocked, current_lv, current_exp, current_gold)
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

def _to_harmony_messages(prompt_messages: list[dict[str, str]]) -> list[Message]:
    """Convert simple role/content dicts to Harmony Message objects."""
    harmony_msgs: list[Message] = []
    first_system = True

    for msg in prompt_messages:
        role, content = msg["role"], msg["content"]

        if role == "system":
            if first_system:
                first_system = False
                # Model-level system message (identity, reasoning config)
                sys_content = (
                    SystemContent.new()
                    .with_reasoning_effort(ReasoningEffort.LOW)
                    .with_conversation_start_date(
                        datetime.datetime.now().strftime("%Y-%m-%d")
                    )
                )
                harmony_msgs.append(
                    Message.from_role_and_content(Role.SYSTEM, sys_content)
                )
                # Custom instructions as developer message
                dev_content = DeveloperContent.new().with_instructions(content)
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
            harmony_msgs.append(
                Message.from_role_and_content(Role.ASSISTANT, content)
            )

    return harmony_msgs


def generate_reply(
    prompt_messages: list[dict[str, str]],
    max_new_tokens: int = 80,
    max_reply_tokens: int = 20,
) -> str:
    """
    Generate a reply.
    max_new_tokens: total budget including reasoning + final content.
    max_reply_tokens: hard cap on final-channel tokens (visible reply).
    """
    t0 = time.perf_counter()

    # Encode conversation with Harmony protocol
    harmony_msgs = _to_harmony_messages(prompt_messages)
    conversation = Conversation.from_messages(harmony_msgs)
    tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    stop_tokens = encoding.stop_tokens_for_assistant_actions()

    # Debug: dump decoded prompt to verify history is included
    decoded_prompt = encoding.decode(tokens)
    logger.info("=== PROMPT DUMP (last 500 chars) ===\n%s", decoded_prompt[-500:])

    t1 = time.perf_counter()

    # Generate via vLLM backend
    parser = StreamableParser(encoding, role=Role.ASSISTANT)
    reply_parts: list[str] = []
    gen_count = 0
    final_token_count = 0
    channels_seen: set[str] = set()

    for predicted_token in generator.generate(
        tokens, stop_tokens=stop_tokens, temperature=0.7, max_tokens=max_new_tokens
    ):
        parser.process(predicted_token)
        gen_count += 1
        if parser.current_channel:
            channels_seen.add(parser.current_channel)
        if parser.last_content_delta and parser.current_channel == "final":
            reply_parts.append(parser.last_content_delta)
            final_token_count += 1
            if final_token_count >= max_reply_tokens:
                break

    t2 = time.perf_counter()

    reply = "".join(reply_parts)
    gen_ms = (t2 - t1) * 1000
    tok_per_sec = gen_count / (t2 - t1) if (t2 - t1) > 0 else 0

    logger.info(
        "generate: input=%d tokens, output=%d tokens (final=%d), "
        "encode=%.1fms, generate=%.1fms (%.1f tok/s), total=%.1fms, "
        "channels=%s, reply_len=%d",
        len(tokens), gen_count, final_token_count,
        (t1 - t0) * 1000, gen_ms, tok_per_sec, (t2 - t0) * 1000,
        channels_seen, len(reply),
    )

    return reply


# =====================================================================
# Endpoints
# =====================================================================

@app.post("/game_config")
async def receive_game_config(cfg: GameConfigRequest):
    """
    Game server pushes zones + zone_enemies tables at startup.
    Persists to nanny_data/environments/ as JSON files with flock.
    All consumers read from these files on each request.
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

    # --- Build zone_enemies dict ---
    new_zone_enemies: dict[int, list[list]] = {}
    for ze in cfg.zone_enemies:
        new_zone_enemies.setdefault(ze.zone_id, []).append(
            [ze.enemy_type, ze.player_lv, ze.count]
        )

    # --- Persist all to JSON files with flock ---
    _flock_write_json(_ENV_ZONES_FILE, raw_zones)
    _flock_write_json(_ENV_ZONE_NAMES_FILE, new_zone_names)
    _flock_write_json(_ENV_ZONE_UNLOCK_FILE, new_zone_unlock)
    _flock_write_json(_ENV_ZONE_ENEMIES_FILE, new_zone_enemies)

    logger.info(
        "game_config: loaded %d zones, %d zone_enemy entries. ZONE_NAMES=%s",
        len(cfg.zones), len(cfg.zone_enemies), new_zone_names,
    )
    return {
        "status": "ok",
        "zones": len(cfg.zones),
        "zone_enemies": len(cfg.zone_enemies),
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

    # Store unlocked zones if provided (accept both snake_case and camelCase)
    uzones = req.unlocked_zones if req.unlocked_zones is not None else req.unlockedZones
    if uzones is not None:
        unlocked_zones[req.user_id] = uzones

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
    reply = generate_reply(prompt_messages)

    # Strip non-latin1 characters (TTS downstream requires latin-1 safe output)
    # Keep printable ASCII + common punctuation; replace CJK/emoji with space then collapse
    raw_reply = reply
    reply = reply.encode("latin-1", errors="ignore").decode("latin-1")
    reply = re.sub(r"[^\x20-\x7E\xa0-\xff]", " ", reply)
    reply = re.sub(r" {2,}", " ", reply).strip()

    # Only store non-empty replies to avoid polluting history with blank assistant turns
    if reply:
        history.append({"role": "assistant", "content": reply})
    else:
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
    parser.add_argument("--context-length", type=int, default=8192,
                        help="Max context length for KV cache (triton backend)")
    args = parser.parse_args()

    # Set up data directories
    global DATA_DIR, SESSIONS_DIR, ENVIRONMENTS_DIR
    global _ENV_ZONES_FILE, _ENV_ZONE_ENEMIES_FILE, _ENV_ZONE_NAMES_FILE, _ENV_ZONE_UNLOCK_FILE, _ENV_ENEMY_STATS_FILE
    DATA_DIR = Path(args.data_dir)
    SESSIONS_DIR = DATA_DIR / "sessions"
    ENVIRONMENTS_DIR = DATA_DIR / "environments"
    _ENV_ZONES_FILE = ENVIRONMENTS_DIR / "zones.json"
    _ENV_ZONE_ENEMIES_FILE = ENVIRONMENTS_DIR / "zone_enemies.json"
    _ENV_ZONE_NAMES_FILE = ENVIRONMENTS_DIR / "zone_names.json"
    _ENV_ZONE_UNLOCK_FILE = ENVIRONMENTS_DIR / "zone_unlock_requirements.json"
    _ENV_ENEMY_STATS_FILE = ENVIRONMENTS_DIR / "enemy_stats.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ENVIRONMENTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Data directory ready. Waiting for game server to POST /game_config.")

    # Load model with vLLM backend
    from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator

    print(f"Loading model: {args.checkpoint} (vLLM backend) ...")
    generator = VLLMGenerator(args.checkpoint, tensor_parallel_size=1)
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    print("Model loaded with vLLM backend.")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
