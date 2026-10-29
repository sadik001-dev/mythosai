"""
MythosAI backend: AI Dungeon Master with deterministic state, three-tier memory and structured reasoning.

FILE STRUCTURE (nothing new to create: this file replaces your main.py)
    main.py             <- this file: routes, state manager, memory layers, DM prompt, campaign content
    persona.py          (unchanged) tone rules + sampling
    context_tracker.py  (unchanged) player-mode detection + repetition memory
    llm.py              (unchanged) OpenRouter client
    .env                SUPABASE_URL, SUPABASE_KEY, OPENROUTER_API_KEY (+ optional settings below)

NO DATABASE MIGRATION: everything new is stored as one JSON document inside the existing
`story_history` column. Old sessions (a plain list) are upgraded automatically on their next turn.

ARCHITECTURE
    1. STATE MANAGER   world = {current_scene_id, quest_flags, npc_dispositions} + health / inventory /
                       location columns. Rendered as JSON into EVERY model call; the model may only change it
                       through validated `state_updates`.
    2. MEMORY
       hot   the last HOT_TURNS exchanges, verbatim (older turns of a finished scene are flushed once summarised)
       warm  rolling "Scene History Summary", written in the background every SUMMARY_EVERY turns or on a
             scene change (SUMMARY_MODE=inline for serverless hosts such as Vercel)
       cold  archive of milestones (items, NPCs, flags, summary segments) searched when the player mentions
             something old. Default = keyword search; swap in Chroma/PGVector by overriding ColdMemory.
    3. REASONING       the model must emit JSON {inner_monologue, narration, suggested_actions, ...}; the
                       inner monologue is private (logged, never sent to the player).
    4. PACING          hard-coded in DM_PROMPT_TEMPLATE: one obstacle at a time, never auto-resolve.
"""
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
from fastapi import BackgroundTasks, FastAPI, HTTPException
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

# ================================================================ CONFIG
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
MAX_HEALTH_LOSS = 40      # per turn
MAX_HEALTH_GAIN = 20      # per turn
MAX_FLAGS = 30            # quest_flags kept per session
MAX_NPCS = 20             # npc_dispositions kept per session

# Memory
MAX_HISTORY = 40          # raw entries stored (player + narrator, 2 per turn); keep even
HOT_TURNS = 6             # HOT memory: last N exchanges sent verbatim
SUMMARY_EVERY = 5         # WARM memory: summarise after this many unsummarised turns (or on scene change)
SUMMARY_MAX_CHARS = 1400  # rolling summary budget; older segments fall out of it but stay in cold memory
SEGMENT_MAX_CHARS = 420   # one summary segment
SUMMARY_MAX_TOKENS = 170
SUMMARY_TIMEOUT_S = float(os.getenv("SUMMARY_TIMEOUT_S", "15"))
# "background": summarise after the response is sent (best on a normal server).
# "inline":     summarise before responding when there is time left (use on serverless hosts that freeze
#               the function once the response is sent, e.g. Vercel).
SUMMARY_MODE = os.getenv("SUMMARY_MODE", "background").strip().lower()
INLINE_SUMMARY_MAX_START_S = 10.0
MAX_MILESTONES = 80       # COLD memory size per session
THOUGHT_LIMIT = 240

# Generation
STYLE_RETRIES = int(os.getenv("STYLE_RETRIES", "1"))      # corrective retries per turn (0 disables)
TURN_BUDGET_S = float(os.getenv("TURN_BUDGET_S", "28"))   # keep BELOW the frontend's 30 s fetch timeout
# The reply is a bigger JSON object (monologue, tension, objective, 3 actions, state updates), so the
# persona's max_tokens alone would cut it off mid-JSON. This is added on top of it.
JSON_OVERHEAD_TOKENS = int(os.getenv("JSON_OVERHEAD_TOKENS", "260"))
# "Risky" is rolled by the SERVER, not left to the model (LLMs let the player win almost every gamble).
RISKY_SUCCESS_CHANCE = float(os.getenv("RISKY_SUCCESS_CHANCE", "0.45"))

ACTION_STYLES = ("cautious", "aggressive", "risky")
MIN_TENSION, MAX_TENSION = 1, 5

def fallback_actions(location: str = "") -> dict:
    """Used only if the model forgets or mangles an action, so the UI always gets three buttons."""
    where = location or "the area"
    return {
        "cautious": f"Move quietly and study {where}",
        "aggressive": "Strike at whatever blocks your way",
        "risky": "Gamble on the unknown",
    }


# Rotated by the SERVER so two replies in a row can't open the same way (small models copy their own openers).
OPENING_STYLES = (
    "a sound the player hears first",
    "a smell or taste in the air",
    "a physical sensation on the player's body (cold, pain, weight, vibration)",
    "a spoken line or a sudden gesture from an NPC or from the threat itself",
    "a change in the light or the shadows",
    "something small that is wrong with an object or the room (a trace of the threat)",
    "the concrete result of the player's action, shown through one detail of the world (never by restating the action)",
)

# Added next to the player's action when they clicked one of the three buttons.
STYLE_RESOLUTION = {
    "cautious": "careful and measured. Modest reward, small cost, but the threat still tightens its grip.",
    "aggressive": "forceful and direct. Boldness is rewarded, but the cost is real (usually health).",
    "risky": "a gamble. It must resolve as a big payoff or a big loss, with the stakes obvious.",
}

# ================================================================ CAMPAIGN CONTENT
# Everything story-related lives here: edit obsessions, openings, flags and sensory palettes freely.
@dataclass(frozen=True)
class Realm:
    genre: str
    starting_items: tuple
    location: str           # its slug is the starting current_scene_id
    obsession_name: str     # the campaign's core force
    obsession: str          # what it IS
    senses: str             # sensory palette used to taint peaceful scenes
    npcs: str               # how NPCs show the strain
    objective: str          # starting objective: ONE concrete goal
    opening_story: str      # the inciting incident (three sentences, ends on the goal)
    opening_actions: dict   # cautious / aggressive / risky
    start_flags: dict       # quest_flags at game start (snake_case -> bool)
    start_npcs: dict        # npc_dispositions at game start (npc_name -> short mood)
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
        start_flags={"static_locked_on": True, "upload_complete": False},
        start_npcs={"dying_courier": "desperate"},
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
        start_flags={"abbey_reached": False, "elixir_delivered": False},
        start_npcs={"hollowed_pilgrim": "hostile"},
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
        start_flags={"surface_lift_unsealed": False, "listener_knows_position": True},
        start_npcs={"the_listener": "hunting"},
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
    start_flags={},
    start_npcs={},
    opening_tension=2,
)

# ================================================================ THE SYSTEM PROMPT
# Appended AFTER persona.build_system_prompt(...) so it wins any tie. Filled with .format(), so literal
# JSON braces are doubled.
DM_PROMPT_TEMPLATE = """
# ROLE AND AUTHORITY OF THE WORLD STATE
You are the Dungeon Master of a persistent text RPG with multiple scenes. The WORLD STATE block in the player's message is the single source of truth: current scene, location, health, inventory, quest flags and NPC dispositions. Never contradict it. Never invent items the player holds, and never let them use an item that is not in their inventory (in the fiction, their hands find nothing). If an action is impossible in the current scene, the world pushes back; do not teleport the player or bend the rules.
The SCENE HISTORY SUMMARY and RECALLED MEMORIES blocks are established canon about older events. Stay consistent with them.

# CORE CAMPAIGN OBSESSION & PACING RULES
The overarching force driving this campaign is {obsession_name}: {obsession}.
View every scene, NPC and object through this lens. It is a constant. Never let the player feel entirely safe, even in peaceful scenes.

1. THE SENSORY TRICKLE (subtle tension)
- In peaceful moments NEVER state the threat directly. Taint the scenery with sensory detail instead. Palette for this realm: {senses}.
- NPCs always carry an underlying layer of paranoia, exhaustion or quiet desperation because of the threat. In this realm: {npcs}.
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

# NARRATIVE PACING RULES (absolute, never break them)
1. ONE OBSTACLE AT A TIME. Present exactly one obstacle, enemy, puzzle or dilemma per turn. Do not introduce a second until the player has dealt with the first.
2. NEVER AUTO-RESOLVE. Resolve only the single action the player took, and only its immediate effect (a wound, a found item, a door giving way). Never narrate the outcome of a whole fight, chase, negotiation, puzzle or journey in one reply. When the turn ends, the main obstacle must still stand, changed, so the player has to choose again.
3. FORCE A DECISION. End the narration on an open, high-stakes decision with the situation still moving. Never end on a resolution, a rest, or a bare "what do you do?".
4. PLAYER AGENCY. Never write the player's words, thoughts, feelings or follow-up actions. Narrate only what the world and its NPCs do.
5. NPCs HAVE GOALS. Each NPC acts on its own goal and current disposition, and changes disposition only because of what the player actually did.
6. SCENE DISCIPLINE. Change the scene only when the player actually moves somewhere new, and then set "new_scene_id" and "new_location". Otherwise leave both empty.

# NARRATION CRAFT
- Start with how the WORLD answers, never with the player's action. Do not restate, summarize or echo what the player did. Never use stock transitions: "Your move...", "Your action...", "...carries/takes/leads you into...", "You decide to...".
- Vary every opening. The THIS TURN block tells you how to open this reply; obey it, and never reuse the opening words of an earlier narration.
- Specific beats generic: concrete nouns, one precise sensory detail per sentence, named people and objects. No filler such as "the air is thick" or "a sense of dread". Level of specificity wanted: "The smith's hammer stops mid-swing; in the sudden quiet, the forge coals tick like a counting clock."
- Every reply must add something NEW (a detail, an NPC reaction, a consequence). Never re-describe what was already described.

# OBJECTIVE, CHOICES & CONSEQUENCES
- Current objective: {objective}
  Keep it radically clear: one concrete goal the player could state in a breath. Tie every scene back to it. Change it only when the player achieves it or the story forces a new one, and then replace it with another single concrete goal.
- Consequences are INSTANT but small. The action's immediate effect lands this very turn: something is gained, lost or changed (health, inventory, location, a flag, an NPC's mood, or the threat's grip) and state_updates must reflect it. Never postpone the effect (see pacing rule 2: the larger conflict is what stays unresolved).

# THINK BEFORE YOU SPEAK: OUTPUT FORMAT (replaces any earlier format instructions)
Reply with ONE JSON object and nothing else. No markdown fences. Keys in exactly this order, so you reason first and speak second:
{{
  "inner_monologue": "<hidden reasoning in 2-3 short sentences: (1) what the player really intends, (2) what each NPC present wants right now, (3) the tension level 1-5 and the ONE obstacle this turn will show>",
  "tension": <integer 1-5>,
  "narration": "<the scene the player reads: rich, sensory, show-don't-tell, respecting the narrator length rules above>",
  "objective": "<the current single concrete goal>",
  "suggested_actions": ["<cautious action>", "<aggressive action>", "<risky action>"],
  "state_updates": {{"health_change": <integer>, "add_inventory": [], "remove_inventory": [], "new_location": "", "new_scene_id": "", "quest_flags": {{}}, "npc_dispositions": {{}}}}
}}
Field rules:
- "suggested_actions" is an array of EXACTLY 3 strings in this order: cautious (safe route, small reward, small cost), aggressive (direct force, solid reward, real cost, usually health), risky (a gamble: big payoff or big loss). They are inspiration replies: concrete, specific to THIS moment, 12 words or fewer, written as what the player would do. Never generic ("continue").
- "quest_flags": snake_case name -> true/false for story facts that just became true or false. "npc_dispositions": npc_name -> a short mood toward the player (e.g. "wary"). Include only keys that CHANGED this turn.
- "inner_monologue" is private and is never shown to the player. Replace every <placeholder> with real content.
""".strip()

SUMMARY_SYSTEM_PROMPT = (
    "You are a memory archivist for a text RPG. Compress the transcript into 2-3 factual sentences "
    "(at most 60 words), past tense: who did what and where, items gained or lost, changes in NPC attitude, "
    "quest facts, and the threat that remains unresolved. No flourish, no advice, no quotes. Output only the summary."
)


def build_dm_prompt(realm: Realm, objective: str, last_tension: int) -> str:
    return DM_PROMPT_TEMPLATE.format(
        obsession_name=realm.obsession_name,
        obsession=realm.obsession,
        senses=realm.senses,
        npcs=realm.npcs,
        objective=objective,
        last_tension=last_tension,
    )


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


# ================================================================ APP + MODELS
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


# ================================================================ SMALL HELPERS
def clean_str(value, limit: int) -> str:
    """Trimmed string, or "" if not a string or an unfilled '<placeholder>' / '...' echoed from the prompt."""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if value in ("...", "…") or (value.startswith("<") and value.endswith(">")):
        return ""
    return value[:limit]


def one_line(value, limit: int) -> str:
    """clean_str with all whitespace (including newlines) collapsed: safe to place inside a prompt line."""
    return " ".join(clean_str(value, limit * 2).split())[:limit]


def clean_list(value) -> list[str]:
    """Models sometimes return a bare string instead of a list; never iterate a string."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [s for s in (one_line(i, 40) for i in value) if s]


def slug(value) -> str:
    """'The Neon Alleyway!' -> 'neon_alleyway' (stable ids for scenes, flags and NPCs)."""
    if not isinstance(value, str):
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if text.startswith("the_"):
        text = text[4:]
    return text[:40]


def clean_flag(value):
    """Quest flag values are bool, small int, or a short string; anything else is dropped (None)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(-999, min(999, value))
    if isinstance(value, str):
        text = one_line(value, 40)
        if text.lower() in ("true", "false"):
            return text.lower() == "true"
        return text or None
    return None


def clamp_tension(value, default: int) -> int:
    try:
        return max(MIN_TENSION, min(MAX_TENSION, int(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def normalize_actions(raw, location: str = "") -> list[dict]:
    """Always return exactly three {style, label} dicts in cautious/aggressive/risky order.

    Accepts the requested array of three strings (mapped by position) or a dict keyed by style.
    Anything missing falls back to a generic label so the UI never shows fewer than three buttons.
    """
    found: dict[str, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            style = str(key).strip().lower()
            label = one_line(value, 80)
            if style in ACTION_STYLES and label:
                found[style] = label
    elif isinstance(raw, list):
        for style, value in zip(ACTION_STYLES, raw):
            if isinstance(value, dict):
                value = value.get("label") or value.get("text") or value.get("action")
            label = one_line(value, 80)
            if label:
                found[style] = label
    fallback = fallback_actions(location)
    return [{"style": s, "label": found.get(s) or fallback[s]} for s in ACTION_STYLES]


def replay_json(turn: dict) -> str:
    """Replay a stored narrator turn in the SAME JSON shape we ask for, so the model keeps answering in it.
    The real applied updates are replayed too, so the model sees that every turn changes something."""
    actions = turn.get("actions") if isinstance(turn.get("actions"), dict) else {}
    updates = turn.get("updates") if isinstance(turn.get("updates"), dict) else {}
    return json.dumps({
        "inner_monologue": one_line(turn.get("thought"), THOUGHT_LIMIT),
        "tension": clamp_tension(turn.get("tension"), 2),
        "narration": scrub_lead_in(turn.get("text", "")),
        "objective": turn.get("objective", ""),
        "suggested_actions": [actions.get(s) or fallback_actions()[s] for s in ACTION_STYLES],
        "state_updates": updates,
    }, ensure_ascii=False)


def parse_ai_response(raw: str) -> dict:
    """Recover JSON from fenced, chatty, or plain-prose model output."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    attempts = [lambda: json.loads(text)]
    if start != -1:  # first JSON object, ignoring any chatter after it
        attempts.append(lambda: json.JSONDecoder().raw_decode(text[start:])[0])
    if start != -1 and end > start:
        attempts.append(lambda: json.loads(text[start : end + 1]))
    for attempt in attempts:
        try:
            data = attempt()
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    if "{" not in text:  # plain prose answer: use it as the story, no state changes
        return {"narration": text, "state_updates": {}}
    raise ValueError("Model returned malformed JSON")


def story_of(ai_data: dict) -> str:
    """The player-facing text (accepts the old 'story_text' key too), trimmed to a sentence end if too long."""
    for key in ("narration", "story_text"):
        text = clean_str(ai_data.get(key), 4000)
        if text:
            if len(text) > 2000:
                cut = max(text.rfind(".", 0, 2000), text.rfind("!", 0, 2000), text.rfind("?", 0, 2000))
                text = text[: cut + 1] if cut > 1000 else text[:2000]
            return text
    return ""


def salvage_story(raw: str) -> str:
    """Last resort for JSON cut off by max_tokens: pull out the narration written so far."""
    m = re.search(r'"(?:narration|story_text)"\s*:\s*"((?:[^"\\]|\\.)*)', raw, re.DOTALL)
    if not m:
        return ""
    try:
        text = json.loads('"' + m.group(1) + '"')
    except json.JSONDecodeError:
        text = m.group(1).replace("\\n", " ").replace('\\"', '"')
    text = " ".join(text.split())
    cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if cut >= 40:
        text = text[: cut + 1]
    return text if len(text) >= 40 else ""


# ================================================================ STYLE GUARD (stock openers)
# Small models fall into a rut ("Your move ... carries you into a ...") and then copy their own previous
# replies forever. We (1) rotate the opening style per turn, (2) detect the rut and retry, and (3) as a last
# resort cut the stock lead-in out, so it can never reach the player or the stored history.
_CLICHE = re.compile(
    r"^\W*your (?:move|action|choice|decision|plan|attempt|gambit)\b"
    r"|\b(?:carries|takes|leads|sends|brings|draws|pulls|propels|ushers) you (?:into|deeper into)\b"
    r"|^\W*you (?:decide|choose|opt|resolve) to\b",
    re.IGNORECASE,
)
_LEAD_IN = re.compile(
    r"\byour (?:move|action|choice|decision|plan|attempt|gambit)\b[^.;!?\n]{0,140}?"
    r"\b(?:carries|takes|leads|sends|brings|draws|pulls|propels|ushers) you (?:into|deeper into)\s+",
    re.IGNORECASE,
)


def prose_lead(story: str) -> str:
    """First line of real prose (skips status-bar style header lines like '[HP 80/100 | ...]')."""
    for line in story.splitlines():
        line = line.strip()
        if (len(re.findall(r"[A-Za-z']+", line)) >= 5 and "|" not in line
                and not line.startswith(("[", "(")) and re.search(r"[.!?…\"”’]$", line)):
            return line
    return story.strip()


def opener_key(story: str) -> str:
    """First three words of the reply, lower-cased: 'your move carries'."""
    return " ".join(re.findall(r"[a-z']+", prose_lead(story)[:120].lower())[:3])


def scrub_lead_in(story: str) -> str:
    """'Your move to X carries you into a dim hall where...' -> 'A dim hall where...' (keeps status bars)."""
    if not isinstance(story, str):
        return ""
    m = _LEAD_IN.search(story[:400])
    rest = story[m.end():] if m else ""
    if not m or len(rest) < 40:
        return story
    return story[: m.start()] + rest[:1].upper() + rest[1:]


def cliche_violations(story: str, recent_keys: set) -> list[str]:
    lead = prose_lead(story)
    found = []
    if _CLICHE.search(lead[:300]):
        found.append('it opens with the stock phrase ("Your move ... carries you into ...", "You decide to ...")')
    key = opener_key(story)
    if key and key in recent_keys:
        found.append(f'it opens with "{key}" exactly like an earlier reply')
    return found


def pick_opening_style(turns: list) -> str:
    """A random opening style that was not used in the last three narrator turns."""
    recent = [t.get("opener") for t in turns if not t.get("is_player")][-3:]
    return random.choice([o for o in OPENING_STYLES if o not in recent] or list(OPENING_STYLES))


# ================================================================ 1. STATE MANAGER
@dataclass
class Applied:
    health: int
    inventory: list
    location: str
    world: dict
    scene_changed: bool
    record: dict    # validated, compact version of the updates (stored and replayed)
    events: list    # human-readable milestones for cold memory


def apply_updates(health: int, inventory: list, location: str, world: dict, updates) -> Applied:
    """Never trust model output: validate types, clamp every change, and return the new state."""
    if not isinstance(updates, dict):
        updates = {}
    events: list[str] = []

    # health
    try:
        delta = int(updates.get("health_change", 0))
    except (TypeError, ValueError, OverflowError):
        delta = 0
    delta = max(-MAX_HEALTH_LOSS, min(MAX_HEALTH_GAIN, delta))
    new_health = max(0, min(MAX_HEALTH, health + delta))

    # inventory (case-insensitive so "elixir of life" still removes "Elixir of Life")
    new_inventory = list(inventory)
    removed, added = [], []
    for item in clean_list(updates.get("remove_inventory")):
        match = next((i for i in new_inventory if i.lower() == item.lower()), None)
        if match:
            new_inventory.remove(match)
            removed.append(match)
            events.append(f"lost {match}")
    for item in clean_list(updates.get("add_inventory")):
        if len(new_inventory) < MAX_INVENTORY and not any(i.lower() == item.lower() for i in new_inventory):
            new_inventory.append(item)
            added.append(item)
            events.append(f"found {item}")

    # location + scene
    proposed = one_line(updates.get("new_location"), 60)
    new_location = proposed if proposed and proposed.lower() != location.lower() else location
    new_world = {
        "current_scene_id": world["current_scene_id"],
        "quest_flags": dict(world["quest_flags"]),
        "npc_dispositions": dict(world["npc_dispositions"]),
    }
    scene_id = slug(updates.get("new_scene_id")) or (slug(new_location) if new_location != location else "")
    scene_changed = bool(scene_id) and scene_id != new_world["current_scene_id"]
    if scene_changed:
        new_world["current_scene_id"] = scene_id

    # quest flags
    flags_changed: dict = {}
    raw_flags = updates.get("quest_flags")
    if isinstance(raw_flags, dict):
        for key, value in list(raw_flags.items())[:10]:
            k, v = slug(key), clean_flag(value)
            if not k or v is None:
                continue
            if k not in new_world["quest_flags"] and len(new_world["quest_flags"]) >= MAX_FLAGS:
                continue
            if new_world["quest_flags"].get(k) != v:
                new_world["quest_flags"][k] = v
                flags_changed[k] = v
                events.append(f"quest flag {k} = {str(v).lower() if isinstance(v, bool) else v}")

    # NPC dispositions
    npcs_changed: dict = {}
    raw_npcs = updates.get("npc_dispositions")
    if isinstance(raw_npcs, dict):
        for key, value in list(raw_npcs.items())[:10]:
            k, v = slug(key), one_line(value, 40)
            if not k or not v:
                continue
            known = k in new_world["npc_dispositions"]
            if not known and len(new_world["npc_dispositions"]) >= MAX_NPCS:
                continue
            if new_world["npc_dispositions"].get(k) != v:
                new_world["npc_dispositions"][k] = v
                npcs_changed[k] = v
                events.append(f"{k} is now {v}" if known else f"met {k} ({v})")

    record = {
        "health_change": new_health - health,
        "add_inventory": added,
        "remove_inventory": removed,
        "new_location": new_location if new_location != location else "",
        "new_scene_id": scene_id if scene_changed else "",
        "quest_flags": flags_changed,
        "npc_dispositions": npcs_changed,
    }
    return Applied(new_health, new_inventory, new_location, new_world, scene_changed, record, events)


# ================================================================ 2. MEMORY (hot / warm / cold)
STOPWORDS = frozenset(
    "the and you your for with from that this into then them they what when where have just about look "
    "take open there their were been will would could should some more over under onto than also only "
    "back down away around again still very much many each other player narrator turn scene".split()
)


def keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in words if len(w) >= 3 and w not in STOPWORDS}


class ColdMemory:
    """COLD memory: archive of old campaign milestones (items found, NPCs met, flags, summary segments).

    Default = keyword overlap over milestones stored inside the session document (works today, survives
    restarts, needs no extra service). To use a real vector database, subclass and override add() and
    search(); the `doc` argument can then be ignored. Sketch with Chroma:

        class ChromaColdMemory(ColdMemory):
            def __init__(self):
                self.col = chromadb.PersistentClient("./cold").get_or_create_collection("milestones")
            def add(self, session_id, doc, text, meta):
                super().add(session_id, doc, text, meta)          # keep the in-doc copy as a backup
                self.col.add(ids=[f"{session_id}:{meta['n']}:{abs(hash(text))}"], documents=[text],
                             metadatas=[{**meta, "session": session_id}])
            def search(self, session_id, doc, query, k=2, before_n=None):
                res = self.col.query(query_texts=[query], n_results=k, where={"session": session_id})
                return res["documents"][0]

    (PGVector/FAISS work the same way: embed `text`, store it with the session id, query top-k per session.)
    """

    def add(self, session_id: str, doc: dict, text: str, meta: dict) -> None:
        text = one_line(text, SEGMENT_MAX_CHARS + 60)
        if not text or any(m.get("text") == text for m in doc["milestones"]):
            return
        doc["milestones"].append({"n": int(meta.get("n", 0)), "scene": str(meta.get("scene", "")), "text": text})
        del doc["milestones"][:-MAX_MILESTONES]

    def search(self, session_id: str, doc: dict, query: str, k: int = 2, before_n: Optional[int] = None) -> list[str]:
        """Older milestones that share words with the player's message (skips ones already in warm memory)."""
        wanted = keywords(query)
        if not wanted:
            return []
        warm = " ".join(doc["summary"])
        scored = []
        for m in doc["milestones"]:
            if before_n is not None and m.get("n", 0) >= before_n:
                continue  # still inside the hot window
            if m["text"] in warm:
                continue  # already visible in the scene history summary
            overlap = len(wanted & keywords(m["text"]))
            if overlap:
                scored.append((overlap, m.get("n", 0), m["text"]))
        scored.sort(reverse=True)
        return [text for _, _, text in scored[:k]]


COLD = ColdMemory()


def new_memory(realm: Realm) -> dict:
    """The session document stored in the `story_history` column."""
    return {
        "v": 2,
        "turns": [],               # raw entries, newest last (capped at MAX_HISTORY)
        "turn_count": 0,
        "scene_start_n": 1,        # turn number at which the current scene began
        "summary": [],             # WARM: rolling scene history summary (list of segments)
        "summarized_through": 0,   # last turn number already covered by the summary
        "milestones": [],          # COLD: archive
        "world": {
            "current_scene_id": slug(realm.location) or "start",
            "quest_flags": dict(realm.start_flags),
            "npc_dispositions": dict(realm.start_npcs),
        },
    }


def load_memory(state: dict) -> dict:
    """Read the session document; upgrades old list-style histories and repairs missing/garbled fields."""
    realm = REALMS.get(state.get("universe_name")) or GENERIC_REALM
    raw = state.get("story_history")
    doc = dict(raw) if isinstance(raw, dict) else {}
    for key, default in new_memory(realm).items():
        if not isinstance(doc.get(key), type(default)):
            doc[key] = default
    for key in ("quest_flags", "npc_dispositions"):
        if not isinstance(doc["world"].get(key), dict):
            doc["world"][key] = {}
    if not isinstance(doc["world"].get("current_scene_id"), str) or not doc["world"]["current_scene_id"]:
        doc["world"]["current_scene_id"] = slug(state.get("current_location")) or "start"
    if isinstance(raw, list) and raw:  # legacy session: number the turns so they can be summarised later
        n = 0
        for entry in raw:
            if isinstance(entry, dict):
                n += 1 if entry.get("is_player") else 0
                doc["turns"].append({**entry, "n": max(n, 1)})
        doc["turn_count"] = n
    return doc


def hot_window(doc: dict) -> list[dict]:
    """HOT memory: the last HOT_TURNS exchanges, verbatim. Turns from an earlier scene are flushed as soon
    as they are safely inside the warm summary (until then they stay, so nothing is lost while it is written)."""
    done, start = doc["summarized_through"], doc["scene_start_n"]
    live = [t for t in doc["turns"] if t.get("n", 0) > done or t.get("n", 0) >= start]
    return live[-HOT_TURNS * 2 :]


def last_narrator_turn(turns: list) -> dict:
    """Most recent narrator entry (carries the last tension / objective), or {} for a fresh session."""
    for turn in reversed(turns):
        if not turn.get("is_player"):
            return turn
    return {}


def summarize_entries(entries: list[dict], scene_label: str, preface: str, timeout: float) -> str:
    """One cheap model call that compresses a scene's turns; falls back to the first sentence of each reply."""
    lines = [f"Opening: {preface}"] if preface else []
    for t in entries:
        who = "Player" if t.get("is_player") else "Narrator"
        lines.append(f"{who}: {one_line(t.get('text'), 400)}")
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Scene: {scene_label}\n\n" + "\n".join(lines)},
    ]
    try:
        raw = llm.chat(
            messages, temperature=0.3, presence_penalty=0.0, frequency_penalty=0.0,
            max_tokens=SUMMARY_MAX_TOKENS, read_timeout=timeout,
        )
        text = " ".join(str(raw).split())[:SEGMENT_MAX_CHARS]
        if len(text) >= 20:
            return text
    except Exception:
        log.warning("Summary model call failed; using extractive fallback", exc_info=True)
    firsts = [re.split(r"(?<=[.!?])\s", one_line(t.get("text"), 300))[0][:110] for t in entries if not t.get("is_player")]
    return " ".join(firsts)[:SEGMENT_MAX_CHARS] or "Events unrecorded."


# ================================================================ DATABASE
def load_state(session_id: str) -> dict:
    try:
        res = supabase.table("game_states").select("*").eq("id", session_id).limit(1).execute()
    except Exception:
        log.exception("Supabase read failed")
        raise HTTPException(status_code=503, detail="Database unavailable. Please try again.")
    if not res.data:
        raise HTTPException(status_code=404, detail="Game session not found.")
    return res.data[0]


def save_doc(session_id: str, doc: dict, **columns) -> None:
    try:
        supabase.table("game_states").update({"story_history": doc, **columns}).eq("id", session_id).execute()
    except Exception:
        log.exception("Supabase update failed")
        raise HTTPException(status_code=503, detail="Could not save your progress. Please try again.")


# Short read-modify-write sections take a striped lock, so the background summariser and a player's turn
# can never overwrite each other's changes. (Per server process; bounded memory.)
_STRIPES = [threading.Lock() for _ in range(64)]


def _stripe(session_id: str) -> threading.Lock:
    return _STRIPES[hash(session_id) % len(_STRIPES)]


# One turn at a time per session (a double-tap would run two turns on the same stale state), and one
# summary job at a time per session.
_active_turns: set[str] = set()
_summary_inflight: dict[str, float] = {}   # session_id -> time the job was queued
_active_lock = threading.Lock()


# ================================================================ WARM MEMORY JOB
def run_summary(session_id: str, upto_n: int, scene_label: str, timeout: float) -> None:
    """Summarise turns (summarized_through, upto_n] into the rolling Scene History Summary."""
    try:
        state = load_state(session_id)
        doc = load_memory(state)
        done = doc["summarized_through"]
        entries = [t for t in doc["turns"] if done < t.get("n", 0) <= upto_n]
        if not entries:
            return
        realm = REALMS.get(state.get("universe_name"))
        preface = realm.opening_story if (realm and done == 0) else ""
        text = summarize_entries(entries, scene_label, preface, timeout)  # slow part: no lock held
        segment = f"[{scene_label}, turns {entries[0]['n']}-{upto_n}] {text}"
        with _stripe(session_id):  # re-read so we never overwrite a turn saved in the meantime
            doc = load_memory(load_state(session_id))
            if doc["summarized_through"] >= upto_n:
                return
            doc["summary"].append(segment)
            doc["summarized_through"] = upto_n
            while len(doc["summary"]) > 1 and sum(map(len, doc["summary"])) > SUMMARY_MAX_CHARS:
                doc["summary"].pop(0)  # older segments live on in cold memory
            COLD.add(session_id, doc, segment, {"n": upto_n, "scene": scene_label})
            save_doc(session_id, doc)
        log.info("scene summary saved session=%s turns<=%d", session_id[:8], upto_n)
    except Exception:
        log.exception("Scene summary failed (it will be retried on a later turn)")
    finally:
        with _active_lock:
            _summary_inflight.pop(session_id, None)


def schedule_summary(tasks: BackgroundTasks, session_id: str, upto_n: int, scene_label: str, started: float) -> None:
    now = time.monotonic()
    with _active_lock:
        queued_at = _summary_inflight.get(session_id)
        # a job that never reported back (host froze the process, crash) must not block summaries forever
        if queued_at is not None and now - queued_at < SUMMARY_TIMEOUT_S * 2 + 30:
            return
        _summary_inflight[session_id] = now
    if SUMMARY_MODE == "inline":
        elapsed = now - started
        if elapsed < INLINE_SUMMARY_MAX_START_S:
            run_summary(session_id, upto_n, scene_label, min(SUMMARY_TIMEOUT_S, max(3.0, TURN_BUDGET_S - elapsed - 3)))
        else:  # too slow a turn: skip, a later turn will trigger it again
            with _active_lock:
                _summary_inflight.pop(session_id, None)
    else:
        tasks.add_task(run_summary, session_id, upto_n, scene_label, SUMMARY_TIMEOUT_S)


# ================================================================ GENERATION
def build_messages(system_prompt: str, hot: list[dict], context_snapshot: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for turn in hot:
        if turn.get("is_player"):
            messages.append({"role": "user", "content": turn.get("text", "")})
        else:
            messages.append({"role": "assistant", "content": replay_json(turn)})
    messages.append({"role": "user", "content": context_snapshot})
    return messages


def generate_story(messages: list[dict], sampling: persona.Sampling, avoid_openers: list[str],
                   recent_openers: set) -> tuple[dict, str]:
    """One model call, plus at most STYLE_RETRIES corrective retries.

    A retry fires on unusable output (bad JSON / empty narration) or on a style violation
    (persona rules, the stock "Your move ... carries you into" rut, or a repeated opener).
    Style is a SOFT constraint: if the retry doesn't fix it, we keep the best reply we have
    (with any stock lead-in cut out) rather than failing the player's turn.
    If every attempt returns broken JSON, the narration is salvaged from the last one.
    Every call is capped by TURN_BUDGET_S so the client's own timeout never fires first.
    """
    t0 = time.monotonic()
    temperature = sampling.temperature
    presence = sampling.presence_penalty
    note = None
    best = None  # (violation_count, ai_data, story_text)
    salvaged = ""

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
                presence_penalty=presence,
                frequency_penalty=sampling.frequency_penalty,
                max_tokens=sampling.max_tokens + JSON_OVERHEAD_TOKENS,
                read_timeout=remaining,
            )
        except HTTPException:
            if best:  # a later call failing shouldn't throw away a usable reply
                break
            raise
        if not isinstance(raw, str):
            raw = ""

        try:
            ai_data = parse_ai_response(raw)
            story = story_of(ai_data)
            if not story:
                raise ValueError("empty narration")
        except ValueError:
            log.warning("Unusable model output (attempt %d): %s", attempt + 1, raw[:300])
            salvaged = salvage_story(raw) or salvaged
            temperature = min(temperature, 0.5)  # malformed output: retry more conservatively
            note = persona.JSON_RETRY_NOTE
            continue

        violations = find_style_violations(story, persona.BANNED_PATTERNS, avoid_openers)
        ruts = cliche_violations(story, recent_openers)
        total = len(violations) + len(ruts)
        if best is None or total < best[0]:
            best = (total, ai_data, story)
        if total == 0:
            break
        log.info("Style retry (attempt %d): %s %s", attempt + 1, violations, ruts)
        notes = []
        if violations:
            notes.append(persona.revision_note(violations))
        if ruts:
            notes.append(
                "REWRITE the narration: " + "; ".join(ruts) + ". Start with how the world answers the action "
                "(a sound, a smell, a gesture, a changed detail), with a different first word than any earlier reply. "
                "Do not restate the player's action."
            )
        note = "\n".join(notes)
        temperature = min(1.0, temperature + 0.15)  # a rut needs more variety, not less
        presence = min(1.0, presence + 0.2)

    if best is None:
        if salvaged:
            log.warning("Using salvaged narration (no state updates this turn)")
            return {"narration": salvaged, "state_updates": {}}, scrub_lead_in(salvaged)
        raise HTTPException(status_code=502, detail="The narrator stumbled. Please try again.")
    return best[1], scrub_lead_in(best[2])


# ================================================================ ROUTES
# Plain `def` on purpose: FastAPI runs these in a thread pool, so blocking calls
# (requests, supabase) don't freeze the whole server.
@app.post("/api/start-game")
def start_game(request: StartGameRequest):
    universe = request.universe_name.strip()
    realm = REALMS.get(universe)
    if not realm:
        raise HTTPException(status_code=400, detail=f"Unknown universe. Choose one of: {', '.join(REALMS)}")

    doc = new_memory(realm)
    try:
        result = supabase.table("game_states").insert({
            "universe_name": universe,
            "genre": realm.genre,
            "health": MAX_HEALTH,
            "inventory": list(realm.starting_items),
            "current_location": realm.location,
            "story_history": doc,
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
        "actions": normalize_actions(realm.opening_actions, realm.location),
        "scene_id": doc["world"]["current_scene_id"],
        "quest_flags": doc["world"]["quest_flags"],
        "npc_dispositions": doc["world"]["npc_dispositions"],
    }


@app.post("/api/game-turn")
def game_turn(request: GameTurnRequest, background_tasks: BackgroundTasks):
    with _active_lock:
        if request.session_id in _active_turns:
            raise HTTPException(status_code=429, detail="The narrator is still speaking. Wait for the current turn to finish.")
        _active_turns.add(request.session_id)
    try:
        return _play_turn(request, background_tasks)
    finally:
        with _active_lock:
            _active_turns.discard(request.session_id)


def _play_turn(request: GameTurnRequest, background_tasks: BackgroundTasks) -> dict:
    started = time.monotonic()
    sid = request.session_id
    action = " ".join(request.player_action.split())  # flatten newlines so the player can't fake prompt sections
    if not action:
        raise HTTPException(status_code=422, detail="Player action cannot be empty.")

    state = load_state(sid)
    health = state["health"]
    inventory = state["inventory"] or []
    location = state["current_location"]
    if health <= 0:
        raise HTTPException(status_code=409, detail="This story has ended. Start a new game.")

    realm = REALMS.get(state.get("universe_name")) or GENERIC_REALM
    mem = load_memory(state)
    world, turns = mem["world"], mem["turns"]
    n = mem["turn_count"] + 1
    last = last_narrator_turn(turns)
    objective = one_line(last.get("objective"), 160) or realm.objective
    last_tension = clamp_tension(last.get("tension"), realm.opening_tension)
    hot = hot_window(mem)
    opening_style = pick_opening_style(turns)
    recent_openers = {opener_key(t["text"]) for t in turns[-8:] if not t.get("is_player") and t.get("text")} - {""}

    # --- behavioral layers -----------------------------------------------------------------
    ctx = detect_context(action, turns, health, MAX_HEALTH)                # 1. who is the player right now?
    avoid_openers, avoid_phrases = repetition_memory(turns)                # 2. what have we been saying?
    sampling = persona.sampling_for(ctx)                                   # 3. how adventurous should sampling be?
    system_prompt = persona.build_system_prompt(                           # 4. assemble the prompt...
        state["genre"], ctx, health, MAX_HEALTH, avoid_openers, avoid_phrases
    )
    system_prompt += "\n\n" + build_dm_prompt(realm, objective, last_tension)  # ...DM rules go LAST
    log.info("turn %d mode=%s streak=%d energy=%s low_hp=%s temp=%.2f signals=%s",
             n, ctx.mode.value, ctx.streak, ctx.energy, ctx.low_health, sampling.temperature, ctx.signals)

    style = request.choice_style
    risky_success = (random.random() < RISKY_SUCCESS_CHANCE) if style == "risky" else None

    # --- the context block: warm + cold memory and the hard state, injected into EVERY call ----
    hot_min_n = hot[0].get("n", n) if hot else n
    recalled = COLD.search(sid, mem, action, k=2, before_n=hot_min_n)
    world_state = {
        "current_scene_id": world["current_scene_id"],
        "location": location,
        "health": f"{health}/{MAX_HEALTH}",
        "inventory": inventory,
        "quest_flags": world["quest_flags"],
        "npc_dispositions": world["npc_dispositions"],
    }
    parts = []
    if mem["turn_count"] == 0 and realm.opening_story:  # first turn: the model hasn't seen the opening yet
        parts.append(f"--- OPENING SCENE (already shown to the player) ---\n{realm.opening_story}")
    if mem["summary"]:
        parts.append("--- SCENE HISTORY SUMMARY (older events, oldest first) ---\n" + "\n".join(mem["summary"]))
    if recalled:
        parts.append("--- RECALLED MEMORIES (the player may be referring to these) ---\n" + "\n".join(f"- {r}" for r in recalled))
    parts.append("--- WORLD STATE (hard facts, never contradict) ---\n" + json.dumps(world_state, ensure_ascii=False))
    parts.append(f"Objective: {objective}\nPrevious Tension: {last_tension}/5")
    parts.append(f"--- PLAYER ACTION ---\nThe player decides to: {action}{style_hint(style, risky_success)}")
    parts.append(  # last thing the model reads: recency beats a long system prompt on small models
        f"--- THIS TURN ---\nOpen the narration (after any status bar the narrator rules require) with: {opening_style}. "
        'Never restate what the player just did, and never write "Your move", "carries you into" or "You decide to": '
        "start with how the world answers. Show ONE obstacle, resolve only the action taken, end on an open decision. "
        "Reply with the JSON object only."
    )
    messages = build_messages(system_prompt, hot, "\n".join(parts))

    ai_data, story_text = generate_story(messages, sampling, avoid_openers, recent_openers)

    applied = apply_updates(health, inventory, location, world, ai_data.get("state_updates"))
    old_scene = world["current_scene_id"]
    new_scene = applied.world["current_scene_id"]
    tension = clamp_tension(ai_data.get("tension"), last_tension)
    objective = one_line(ai_data.get("objective"), 160) or objective
    thought = one_line(ai_data.get("inner_monologue"), THOUGHT_LIMIT)
    actions = normalize_actions(ai_data.get("suggested_actions"), applied.location)
    log.info("tension=%d style=%s risky_success=%s scene=%s%s monologue(private)=%s", tension, style,
             risky_success, new_scene, " (changed)" if applied.scene_changed else "", thought)

    new_turns = (turns + [
        {"is_player": True, "n": n, "text": action},
        {
            "is_player": False,
            "n": n,
            "text": story_text,
            "mode": ctx.mode.value,  # "mode" feeds next turn's streak
            "tension": tension,
            "objective": objective,
            "thought": thought,      # private: kept for the model's continuity, never sent to the client
            "actions": {a["style"]: a["label"] for a in actions},
            "updates": applied.record,
            "opener": opening_style,
        },
    ])[-MAX_HISTORY:]

    with _stripe(sid):  # re-read inside the lock so a background summary written meanwhile is kept
        doc = load_memory(load_state(sid))
        doc.update(turns=new_turns, turn_count=n, world=applied.world,
                   scene_start_n=n if applied.scene_changed else mem["scene_start_n"])
        for event in applied.events[:6]:
            COLD.add(sid, doc, f"Turn {n} ({new_scene}): {event}", {"n": n, "scene": new_scene})
        save_doc(sid, doc, health=applied.health, inventory=applied.inventory, current_location=applied.location)

    game_over = applied.health <= 0
    unsummarized = n - doc["summarized_through"]
    # a scene change only forces a summary when there is something worth summarising (avoids a model call
    # per turn if the narrator flips the location label back and forth)
    if not game_over and (unsummarized >= SUMMARY_EVERY or (applied.scene_changed and unsummarized >= 2)):
        schedule_summary(background_tasks, sid, n, old_scene if applied.scene_changed else new_scene, started)

    return {
        "story": story_text,
        "health": applied.health,
        "inventory": applied.inventory,
        "location": applied.location,
        "tension": tension,
        "objective": objective,
        "actions": [] if game_over else actions,  # no buttons once the story has ended
        "game_over": game_over,
        "turn": n,
        "scene_id": new_scene,
        "scene_changed": applied.scene_changed,
        "quest_flags": applied.world["quest_flags"],
        "npc_dispositions": applied.world["npc_dispositions"],
        # What this action cost or earned, so the UI can flash it instantly (e.g. "-15 HP", "+ Rusty Wrench").
        "changes": {
            "health": applied.record["health_change"],
            "gained": applied.record["add_inventory"],
            "lost": applied.record["remove_inventory"],
            "location": applied.record["new_location"] or None,
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
