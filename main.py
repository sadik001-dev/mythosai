import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from supabase import Client, create_client

import persona
from context_tracker import detect_context, find_style_violations, repetition_memory
from llm import OpenRouterClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mythosai")

# ---------------------------------------------------------------- config
REQUIRED = ("SUPABASE_URL", "SUPABASE_KEY", "OPENROUTER_API_KEY")
missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing required .env values: {', '.join(missing)}")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]  # server-side only; never ship this to the browser
llm = OpenRouterClient(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free"),
)
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",")
    if o.strip()
]

# Gameplay limits
MAX_HEALTH = 100
MAX_INVENTORY = 20
MAX_HISTORY = 40          # stored entries (player/narrator pairs, keep even)
CONTEXT_ENTRIES = 6       # entries sent back to the model each turn
MAX_HEALTH_LOSS = 40      # per turn
MAX_HEALTH_GAIN = 20      # per turn

# Generation limits
STYLE_RETRIES = int(os.getenv("STYLE_RETRIES", "1"))      # corrective retries per turn (0 disables)
TURN_BUDGET_S = float(os.getenv("TURN_BUDGET_S", "28"))   # keep BELOW the frontend's 30 s fetch timeout

# The reply is now a bigger JSON object (inner_monologue, tension, objective, 3 actions...),
# so the persona's max_tokens alone would cut it off mid-JSON. This is added on top of it.
JSON_OVERHEAD_TOKENS = int(os.getenv("JSON_OVERHEAD_TOKENS", "200"))
# "Risky" is rolled by the SERVER, not left to the model (LLMs let the player win almost every gamble).
RISKY_SUCCESS_CHANCE = float(os.getenv("RISKY_SUCCESS_CHANCE", "0.45"))

# ================================================================ CAMPAIGN CONTENT
# Everything story-related lives here: edit the obsessions, openings and sensory palettes freely.
ACTION_STYLES = ("cautious", "aggressive", "risky")
MIN_TENSION, MAX_TENSION = 1, 5

# Used only if the model forgets or mangles an action, so the UI always gets three buttons.
FALLBACK_ACTIONS = {
    "cautious": "Proceed carefully",
    "aggressive": "Strike first",
    "risky": "Take the gamble",
}

# Added next to the player's action when they clicked one of the three buttons.
STYLE_RESOLUTION = {
    "cautious": "careful and measured. Modest reward, small cost, but the threat still tightens its grip.",
    "aggressive": "forceful and direct. Boldness is rewarded, but the cost is real (usually health).",
    "risky": "a gamble. It must resolve as a big payoff or a big loss, with the stakes obvious.",
}


@dataclass(frozen=True)
class Realm:
    genre: str
    starting_items: tuple
    location: str
    obsession_name: str     # the campaign's core force
    obsession: str          # what it IS
    senses: str             # sensory palette used to taint peaceful scenes
    npcs: str               # how NPCs show the strain
    objective: str          # starting objective: ONE concrete goal
    opening_story: str      # the inciting incident (three sentences, ends on the goal)
    opening_actions: dict   # cautious / aggressive / risky
    opening_tension: int = 3


# Server-side whitelist: the client can never inject its own "genre" or story into the prompt.
REALMS: dict[str, Realm] = {
    "Neo-Tokyo 2099": Realm(
        genre="Sci-Fi / Cyberpunk",
        starting_items=("Data-Deck", "50 Credits"),
        location="Neon Alleyway",
        obsession_name="The Static",
        obsession=(
            "a relentless mechanical plague, a hunting intelligence that spreads through every "
            "screen, implant and machine, and has locked onto the player's neural signature"
        ),
        senses=(
            "a hot-metal and ozone taste in the air; neon signs that skip a frame; a low rhythmic "
            "hum behind the walls; reflections that lag half a second behind the player"
        ),
        npcs=(
            "twitching hands, implants that glitch mid-sentence, people who go silent when a "
            "screen turns toward them, eyes that flicker to every camera"
        ),
        objective="Reach the Spire uplink at the end of Rust Street and upload the Data-Deck.",
        opening_story=(
            "Your Data-Deck shrieks awake in your hand, and every dead screen in the Neon Alleyway "
            "swivels to stare at you - the Static has locked onto your neural signature, and the air "
            "tastes of hot metal. A dying courier slumps against the wall, rasping \"Spire uplink... "
            "it can't follow you past the old firewall\" as white noise floods his eyes. "
            "Your goal: reach the Spire uplink at the end of Rust Street and upload the Deck before "
            "the Static catches you."
        ),
        opening_actions={
            "cautious": "Slip into the maintenance duct and crawl toward the Spire",
            "aggressive": "Smash the nearest screen and sprint down Rust Street",
            "risky": "Jack the Deck into the courier's implant to steal his route",
        },
    ),
    "Elfridge Chronicles": Realm(
        genre="Grimdark / Fantasy",
        starting_items=("Iron Sword", "Elixir of Life"),
        location="Misty Crossroads",
        obsession_name="The Hollowing",
        obsession=(
            "a creeping corruption rising from the roots of the world, hollowing the living into "
            "whispering husks, and it has marked the player"
        ),
        senses=(
            "a sweet grave-soil rot; mist that drifts against the wind; birdsong that stops "
            "mid-note; roots that tighten under the road; shadows that stretch a little too long"
        ),
        npcs=(
            "bloodshot, sleepless eyes; villagers who count the doors twice; soldiers who flinch "
            "at their own shadows; hands gripping lucky charms too hard; whispers that stop when "
            "the player approaches"
        ),
        objective="Reach Saint Veyra's Abbey by nightfall and deliver the Elixir of Life.",
        opening_story=(
            "The mist at the Misty Crossroads parts to reveal a dead pilgrim kneeling in the road, "
            "black roots twitching from his mouth toward the Elixir of Life at your belt. Someone has "
            "gouged one word into the Abbey signpost - RUN - and the trees behind you lean closer than "
            "they did a moment ago. Your goal: reach Saint Veyra's Abbey by nightfall and place the "
            "Elixir in the hands of its last healer."
        ),
        opening_actions={
            "cautious": "Circle wide through the thornwood to bypass the pilgrim",
            "aggressive": "Cut down the pilgrim with your Iron Sword before the roots move",
            "risky": "Pour a drop of the Elixir on the roots to see if it burns them",
        },
    ),
    "Vault 404": Realm(
        genre="Post-Apocalyptic",
        starting_items=("Ration Cube", "Flashlight"),
        location="Blast Door Antechamber",
        obsession_name="The Listener",
        obsession=(
            "an omniscient entity inside the Vault's walls that hunts the last survivor by sound "
            "and warmth, and always seems to know exactly where they are"
        ),
        senses=(
            "slow knocking in the pipes; lights that dim as the player passes; warm breath from "
            "sealed vents; footsteps that echo half a second late; a smell of wet iron"
        ),
        npcs=(
            "survivors who whisper and ration their words, sleep with their eyes open, or tap the "
            "walls in patterns when they think no one is watching"
        ),
        objective="Reach the Level 3 override terminal and unseal the surface lift.",
        opening_story=(
            "The lights in the Blast Door Antechamber die one by one, and from the sealed vents comes a "
            "slow, patient knocking that matches your heartbeat. The terminal beside you flickers a "
            "single line - WE SEE YOU, 404 - as the door behind you grinds shut and locks. Your goal: "
            "reach the Level 3 override terminal and unseal the surface lift before the knocking reaches you."
        ),
        opening_actions={
            "cautious": "Switch off the Flashlight and creep along the wall in the dark",
            "aggressive": "Smash the terminal's camera and force the inner door",
            "risky": "Answer the knocking by tapping back on the vent",
        },
    ),
}

# Safety net for old or unknown sessions (never used to start a new game).
GENERIC_REALM = Realm(
    genre="Dark Adventure",
    starting_items=(),
    location="Unknown",
    obsession_name="The Dread",
    obsession="an unnamed, creeping dread that hunts the player and taints everything it touches",
    senses="a wrong smell in the air; shadows that stretch against the light; a faint rhythmic hum",
    npcs="trembling hands, sleepless eyes, and a paranoia they cannot explain",
    objective="Survive and find a way forward.",
    opening_story="",
    opening_actions={},
    opening_tension=2,
)


def build_campaign_prompt(realm: Realm, objective: str, last_tension: int) -> str:
    """Pacing rules + output contract. Appended AFTER the persona prompt so it wins any tie."""
    return f"""
# CORE CAMPAIGN OBSESSION & PACING RULES
The overarching force driving this campaign is {realm.obsession_name}: {realm.obsession}.
View every scene, NPC and object through this lens. It is a constant. Never let the player feel entirely safe, even in peaceful scenes.

1. THE SENSORY TRICKLE (subtle tension)
- In peaceful moments NEVER state the threat directly. Taint the scenery with sensory detail instead. Palette for this realm: {realm.senses}.
- NPCs always carry an underlying layer of paranoia, exhaustion or quiet desperation because of the threat. In this realm: {realm.npcs}.
- Every NPC, item and room carries at least one small trace of it.

2. NARRATIVE ESCALATION SCALE
Each turn, FIRST rate the scene's Tension Level (1-5) in "inner_monologue", then set "tension" to match and write the scene at that level:
- 1-2 (exploration / safe zones): the threat is purely atmospheric, a background detail. The player feels uneasy but safe to act.
- 3-4 (conflict / choice zones): the threat actively interferes. A path is corrupted, an NPC snaps under the pressure, or an item acts strangely, forcing a difficult decision.
- 5 (climax / encounter): the threat manifests directly. The danger is immediate, violent or psychologically overwhelming.
The previous turn was Tension {last_tension}. Move about one level per turn unless the player's action clearly justifies a bigger jump, and let a quieter beat follow a climax so the next spike hits harder.

3. STRICT NARRATIVE BREAKS
- NEVER resolve the core threat. It is an inescapable loop. If the player tries to destroy or fix it, the environment pushes back: reveal a deeper layer of the nightmare and force them forward into the next scene.
- Show, don't tell. Never say the player is scared. Describe the chilling weight of the room and let them feel it.

# OBJECTIVE, CHOICES & CONSEQUENCES
- Current objective: {objective}
  Keep it radically clear: one concrete goal the player could state in a breath. Tie every scene back to it. Change it only when the player achieves it or the story forces a new one, and then replace it with another single concrete goal.
- Consequences are INSTANT. The player's action resolves in this very turn: something is gained, lost or changed (health, inventory, location, or the threat's grip) and state_updates must reflect it. Never postpone the result.
- Give exactly three next actions in "suggested_actions", each fitted to the CURRENT situation (never generic like "continue"), 12 words or fewer:
  cautious: the safe route. Small reward, small cost.
  aggressive: direct force. Solid reward, real cost (usually health).
  risky: a gamble. Big payoff or big loss.

# OUTPUT FORMAT (replaces any earlier format instructions)
Reply with ONE JSON object and nothing else. No markdown fences. Keys in this order:
{{
  "inner_monologue": "<one short sentence: the tension level and how the threat taints this scene>",
  "tension": <integer 1-5>,
  "story_text": "<the scene in the narrator's voice, respecting the narrator length rules above>",
  "objective": "<the current single concrete goal>",
  "suggested_actions": {{"cautious": "<action>", "aggressive": "<action>", "risky": "<action>"}},
  "state_updates": {{"health_change": <integer>, "add_inventory": [], "remove_inventory": [], "new_location": ""}}
}}
Replace every <placeholder> with real content. "inner_monologue" is private and is never shown to the player.
""".strip()


def style_hint(style, risky_success=None) -> str:
    """Extra lines appended to the player's action when they clicked a button."""
    if style not in STYLE_RESOLUTION:
        return ""
    hint = f"\n(The player chose the {style.upper()} approach: {STYLE_RESOLUTION[style]})"
    if risky_success is True:
        hint += "\nFATE HAS DECIDED: the gamble SUCCEEDS. Reward the player visibly, but the threat still tightens its grip."
    elif risky_success is False:
        hint += "\nFATE HAS DECIDED: the gamble FAILS. Show the cost immediately and make health or inventory reflect it."
    return hint


# ---------------------------------------------------------------- app + models
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI(title="MythosAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # no cookies are used
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class StartGameRequest(BaseModel):
    universe_name: str = Field(min_length=1, max_length=60)


class GameTurnRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    player_action: str = Field(min_length=1, max_length=300)
    # Set by the frontend when the player clicked one of the three buttons; omit for typed actions.
    choice_style: Optional[Literal["cautious", "aggressive", "risky"]] = None


# ---------------------------------------------------------------- helpers
def clean_str(value, limit: int) -> str:
    """Trimmed string, or "" if not a string or an unfilled '<placeholder>' / '...' echoed from the prompt."""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if value in ("...", "…") or (value.startswith("<") and value.endswith(">")):
        return ""
    return value[:limit]


def clean_list(value) -> list[str]:
    """Models sometimes return a bare string instead of a list; never iterate a string."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [s for s in (clean_str(i, 40) for i in value) if s]


def clamp_tension(value, default: int) -> int:
    try:
        return max(MIN_TENSION, min(MAX_TENSION, int(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def normalize_actions(raw) -> list[dict]:
    """Always return exactly three {style, label} dicts in cautious/aggressive/risky order.

    Accepts the requested dict, or a bare list of three strings (mapped by position).
    Anything missing falls back to a generic label so the UI never shows fewer than three buttons.
    """
    found: dict[str, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            style = str(key).strip().lower()
            label = " ".join(clean_str(value, 80).split())
            if style in ACTION_STYLES and label:
                found[style] = label
    elif isinstance(raw, list):
        for style, value in zip(ACTION_STYLES, raw):
            if isinstance(value, dict):
                value = value.get("label") or value.get("text") or value.get("action")
            label = " ".join(clean_str(value, 80).split())
            if label:
                found[style] = label
    return [{"style": s, "label": found.get(s) or FALLBACK_ACTIONS[s]} for s in ACTION_STYLES]


def replay_json(turn: dict) -> str:
    """Replay a stored narrator turn in the SAME JSON shape we ask for, so the model keeps answering in it."""
    actions = turn.get("actions") if isinstance(turn.get("actions"), dict) else FALLBACK_ACTIONS
    return json.dumps({
        "inner_monologue": turn.get("thought", ""),
        "tension": clamp_tension(turn.get("tension"), 2),
        "story_text": turn["text"],
        "objective": turn.get("objective", ""),
        "suggested_actions": actions,
        "state_updates": {},
    })


def parse_ai_response(raw: str) -> dict:
    """Recover JSON from fenced, chatty, or plain-prose model output."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    if "{" not in text:  # plain prose answer: use it as the story, no state changes
        return {"story_text": text, "state_updates": {}}
    raise ValueError("Model returned malformed JSON")


def apply_updates(health: int, inventory: list, location: str, updates) -> tuple[int, list, str]:
    """Never trust model output: validate types and clamp every change."""
    if not isinstance(updates, dict):
        updates = {}
    try:
        delta = int(updates.get("health_change", 0))
    except (TypeError, ValueError, OverflowError):
        delta = 0
    delta = max(-MAX_HEALTH_LOSS, min(MAX_HEALTH_GAIN, delta))
    new_health = max(0, min(MAX_HEALTH, health + delta))

    new_inventory = list(inventory)
    for item in clean_list(updates.get("remove_inventory")):
        if item in new_inventory:
            new_inventory.remove(item)
    for item in clean_list(updates.get("add_inventory")):
        if item not in new_inventory and len(new_inventory) < MAX_INVENTORY:
            new_inventory.append(item)

    new_location = clean_str(updates.get("new_location"), 60) or location
    return new_health, new_inventory, new_location


def last_narrator_turn(history: list) -> dict:
    """Most recent narrator entry (carries the last tension / objective), or {} for a fresh session."""
    for turn in reversed(history):
        if not turn.get("is_player"):
            return turn
    return {}


def load_state(session_id: str) -> dict:
    try:
        res = supabase.table("game_states").select("*").eq("id", session_id).limit(1).execute()
    except Exception:
        log.exception("Supabase read failed")
        raise HTTPException(status_code=503, detail="Database unavailable. Please try again.")
    if not res.data:
        raise HTTPException(status_code=404, detail="Game session not found.")
    return res.data[0]


def build_messages(system_prompt: str, history: list, context_snapshot: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-CONTEXT_ENTRIES:]:
        if turn.get("is_player"):
            messages.append({"role": "user", "content": turn["text"]})
        else:
            messages.append({"role": "assistant", "content": replay_json(turn)})
    messages.append({"role": "user", "content": context_snapshot})
    return messages


def generate_story(messages: list[dict], sampling: persona.Sampling, avoid_openers: list[str]) -> tuple[dict, str]:
    """One model call, plus at most STYLE_RETRIES corrective retries.

    A retry fires on unusable output (bad JSON / empty story) or on a style violation
    (boilerplate, repeated openers). Style is a SOFT constraint: if the retry doesn't
    fix it, we keep the best reply we have rather than failing the player's turn.
    Every call is capped by TURN_BUDGET_S so the client's own timeout never fires first.
    """
    t0 = time.monotonic()
    temperature = sampling.temperature
    note = None
    best = None  # (violation_count, ai_data, story_text)

    for attempt in range(STYLE_RETRIES + 1):
        remaining = TURN_BUDGET_S - (time.monotonic() - t0)
        if attempt > 0 and remaining < 8:
            break

        msgs = messages
        if note:  # attach the correction to the final user message; avoids a mid-chat system role
            msgs = messages[:-1] + [{"role": "user", "content": messages[-1]["content"] + "\n\n" + note}]

        try:
            raw = llm.chat(
                msgs,
                temperature=temperature,
                presence_penalty=sampling.presence_penalty,
                frequency_penalty=sampling.frequency_penalty,
                max_tokens=sampling.max_tokens + JSON_OVERHEAD_TOKENS,
                read_timeout=remaining,
            )
        except HTTPException:
            if best:  # a later call failing shouldn't throw away a usable reply
                break
            raise

        try:
            ai_data = parse_ai_response(raw)
            story = clean_str(ai_data.get("story_text"), 2000)
            if not story:
                raise ValueError("empty story_text")
        except ValueError:
            log.warning("Unusable model output (attempt %d): %s", attempt + 1, raw[:300])
            temperature = min(temperature, 0.5)  # malformed output: retry more conservatively
            note = persona.JSON_RETRY_NOTE
            continue

        violations = find_style_violations(story, persona.BANNED_PATTERNS, avoid_openers)
        if best is None or len(violations) < best[0]:
            best = (len(violations), ai_data, story)
        if not violations:
            break
        log.info("Style retry (attempt %d): %s", attempt + 1, violations)
        note = persona.revision_note(violations)

    if best is None:
        raise HTTPException(status_code=502, detail="The narrator stumbled. Please try again.")
    return best[1], best[2]


# One turn at a time per session: a double-tap would otherwise run two turns on the same stale
# state and the second save would silently overwrite the first. (Per server process.)
_active_turns: set[str] = set()
_active_lock = threading.Lock()


# ---------------------------------------------------------------- routes
# Plain `def` on purpose: FastAPI runs these in a thread pool, so blocking calls
# (requests, supabase) don't freeze the whole server.
@app.post("/api/start-game")
def start_game(request: StartGameRequest):
    universe = request.universe_name.strip()
    realm = REALMS.get(universe)
    if not realm:
        raise HTTPException(status_code=400, detail=f"Unknown universe. Choose one of: {', '.join(REALMS)}")

    try:
        result = supabase.table("game_states").insert({
            "universe_name": universe,
            "genre": realm.genre,
            "health": MAX_HEALTH,
            "inventory": list(realm.starting_items),
            "current_location": realm.location,
            "story_history": [],
        }).execute()
    except Exception:
        log.exception("Supabase insert failed")
        raise HTTPException(status_code=503, detail="Could not start a new game. Please try again.")

    if not result.data:
        raise HTTPException(status_code=503, detail="Could not start a new game. Please try again.")
    row = result.data[0]
    # The inciting incident is authored: instant, free, and it always ends on a crystal-clear goal.
    return {
        "session_id": row["id"],
        "health": row["health"],
        "inventory": row["inventory"],
        "location": row["current_location"],
        "story": realm.opening_story,
        "objective": realm.objective,
        "tension": realm.opening_tension,
        "actions": normalize_actions(realm.opening_actions),
    }


@app.post("/api/game-turn")
def game_turn(request: GameTurnRequest):
    with _active_lock:
        if request.session_id in _active_turns:
            raise HTTPException(status_code=429, detail="The narrator is still speaking. Wait for the current turn to finish.")
        _active_turns.add(request.session_id)
    try:
        return _play_turn(request)
    finally:
        with _active_lock:
            _active_turns.discard(request.session_id)


def _play_turn(request: GameTurnRequest) -> dict:
    action = " ".join(request.player_action.split())  # flatten newlines so the player can't fake prompt sections
    if not action:
        raise HTTPException(status_code=422, detail="Player action cannot be empty.")

    state = load_state(request.session_id)
    health = state["health"]
    inventory = state["inventory"] or []
    location = state["current_location"]
    history = state["story_history"] or []

    if health <= 0:
        raise HTTPException(status_code=409, detail="This story has ended. Start a new game.")

    # --- campaign state (derived from stored history, so no new DB columns are needed) ----
    realm = REALMS.get(state.get("universe_name")) or GENERIC_REALM
    last = last_narrator_turn(history)
    objective = clean_str(last.get("objective"), 160) or realm.objective
    last_tension = clamp_tension(last.get("tension"), realm.opening_tension)

    # --- behavioral layers ---------------------------------------------------------------
    ctx = detect_context(action, history, health, MAX_HEALTH)              # 1. who is the player right now?
    avoid_openers, avoid_phrases = repetition_memory(history)              # 2. what have we been saying?
    sampling = persona.sampling_for(ctx)                                   # 3. how adventurous should sampling be?
    system_prompt = persona.build_system_prompt(                           # 4. assemble the prompt...
        state["genre"], ctx, health, MAX_HEALTH, avoid_openers, avoid_phrases
    )
    system_prompt += "\n\n" + build_campaign_prompt(realm, objective, last_tension)  # ...campaign rules go LAST
    log.info("turn mode=%s streak=%d energy=%s low_hp=%s temp=%.2f signals=%s",
             ctx.mode.value, ctx.streak, ctx.energy, ctx.low_health, sampling.temperature, ctx.signals)

    style = request.choice_style
    risky_success = (random.random() < RISKY_SUCCESS_CHANCE) if style == "risky" else None

    # On the very first turn the model has no history, so show it the opening scene the player just read.
    opening_block = ""
    if not history and realm.opening_story:
        opening_block = f"--- OPENING SCENE (already shown to the player) ---\n{realm.opening_story}\n"

    context_snapshot = (
        f"{opening_block}"
        f"--- CURRENT STATE ---\n"
        f"Location: {location}\n"
        f"Current Health: {health}/{MAX_HEALTH}\n"
        f"Current Inventory: {json.dumps(inventory)}\n"
        f"Objective: {objective}\n"
        f"Previous Tension: {last_tension}/5\n"
        f"--- PLAYER ACTION ---\n"
        f"The player decides to: {action}"
        f"{style_hint(style, risky_success)}"
    )
    messages = build_messages(system_prompt, history, context_snapshot)

    ai_data, story_text = generate_story(messages, sampling, avoid_openers)

    new_health, new_inventory, new_location = apply_updates(
        health, inventory, location, ai_data.get("state_updates")
    )

    tension = clamp_tension(ai_data.get("tension"), last_tension)
    objective = clean_str(ai_data.get("objective"), 160) or objective
    thought = clean_str(ai_data.get("inner_monologue"), 200)
    actions = normalize_actions(ai_data.get("suggested_actions"))
    log.info("tension=%d style=%s risky_success=%s monologue(private)=%s", tension, style, risky_success, thought)

    history = (history + [
        {"is_player": True, "text": action},
        {
            "is_player": False,
            "text": story_text,
            "mode": ctx.mode.value,  # "mode" feeds next turn's streak
            "tension": tension,
            "objective": objective,
            "thought": thought,  # private: stored for the model's continuity, never sent to the client
            "actions": {a["style"]: a["label"] for a in actions},
        },
    ])[-MAX_HISTORY:]

    try:
        supabase.table("game_states").update({
            "health": new_health,
            "inventory": new_inventory,
            "current_location": new_location,
            "story_history": history,
        }).eq("id", request.session_id).execute()
    except Exception:
        log.exception("Supabase update failed")
        raise HTTPException(status_code=503, detail="Could not save your progress. Please try again.")

    game_over = new_health <= 0
    return {
        "story": story_text,
        "health": new_health,
        "inventory": new_inventory,
        "location": new_location,
        "tension": tension,
        "objective": objective,
        "actions": [] if game_over else actions,  # no buttons once the story has ended
        "game_over": game_over,
        # What this action cost or earned, so the UI can flash it instantly (e.g. "-15 HP", "+ Rusty Wrench").
        "changes": {
            "health": new_health - health,
            "gained": [i for i in new_inventory if i not in inventory],
            "lost": [i for i in inventory if i not in new_inventory],
            "location": new_location if new_location != location else None,
        },
    }


# Optional: put index.html in a "static" folder next to main.py and the frontend is
# served from the same origin (no CORS needed, relative /api/... calls just work).
# Keep this AFTER the API routes.
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
