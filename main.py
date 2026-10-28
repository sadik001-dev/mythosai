import json
import logging
import os
import re
import time
from pathlib import Path

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

# v1.1 generation limits
STYLE_RETRIES = int(os.getenv("STYLE_RETRIES", "1"))      # corrective retries per turn (0 disables)
TURN_BUDGET_S = float(os.getenv("TURN_BUDGET_S", "28"))   # keep BELOW the frontend's 30 s fetch timeout

# Server-side whitelist: the client can no longer inject its own "genre" into the prompt.
REALMS = {
    "Neo-Tokyo 2099": {
        "genre": "Sci-Fi / Cyberpunk",
        "items": ["Data-Deck", "50 Credits"],
        "location": "Neon Alleyway",
    },
    "Elfridge Chronicles": {
        "genre": "Grimdark / Fantasy",
        "items": ["Iron Sword", "Elixir of Life"],
        "location": "Misty Crossroads",
    },
    "Vault 404": {
        "genre": "Post-Apocalyptic",
        "items": ["Ration Cube", "Flashlight"],
        "location": "Blast Door Antechamber",
    },
}

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI(title="MythosAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # no cookies are used
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------- models
class StartGameRequest(BaseModel):
    universe_name: str = Field(min_length=1, max_length=60)


class GameTurnRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    player_action: str = Field(min_length=1, max_length=300)


# ---------------------------------------------------------------- helpers
def clean_str(value, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def clean_list(value) -> list[str]:
    """Models sometimes return a bare string instead of a list; never iterate a string."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [s for s in (clean_str(i, 40) for i in value) if s]


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
        else:  # replay narrator turns in the JSON shape we ask for, so the model keeps answering in JSON
            messages.append({
                "role": "assistant",
                "content": json.dumps({"story_text": turn["text"], "state_updates": {}}),
            })
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
                max_tokens=sampling.max_tokens,
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


# ---------------------------------------------------------------- routes
# Plain `def` on purpose: FastAPI runs these in a thread pool, so blocking calls
# (requests, supabase) no longer freeze the whole server.
@app.post("/api/start-game")
def start_game(request: StartGameRequest):
    realm = REALMS.get(request.universe_name.strip())
    if not realm:
        raise HTTPException(status_code=400, detail=f"Unknown universe. Choose one of: {', '.join(REALMS)}")

    try:
        result = supabase.table("game_states").insert({
            "universe_name": request.universe_name.strip(),
            "genre": realm["genre"],
            "health": MAX_HEALTH,
            "inventory": realm["items"],
            "current_location": realm["location"],
            "story_history": [],
        }).execute()
    except Exception:
        log.exception("Supabase insert failed")
        raise HTTPException(status_code=503, detail="Could not start a new game. Please try again.")

    if not result.data:
        raise HTTPException(status_code=503, detail="Could not start a new game. Please try again.")
    row = result.data[0]
    return {
        "session_id": row["id"],
        "health": row["health"],
        "inventory": row["inventory"],
        "location": row["current_location"],
    }


@app.post("/api/game-turn")
def game_turn(request: GameTurnRequest):
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

    # --- v1.1 behavioral layers --------------------------------------------
    ctx = detect_context(action, history, health, MAX_HEALTH)              # 1. who is the player right now?
    avoid_openers, avoid_phrases = repetition_memory(history)              # 2. what have we been saying?
    sampling = persona.sampling_for(ctx)                                   # 3. how adventurous should sampling be?
    system_prompt = persona.build_system_prompt(                           # 4. assemble the prompt
        state["genre"], ctx, health, MAX_HEALTH, avoid_openers, avoid_phrases
    )
    log.info("turn mode=%s streak=%d energy=%s low_hp=%s temp=%.2f signals=%s",
             ctx.mode.value, ctx.streak, ctx.energy, ctx.low_health, sampling.temperature, ctx.signals)

    context_snapshot = (
        f"--- CURRENT STATE ---\n"
        f"Location: {location}\n"
        f"Current Health: {health}/{MAX_HEALTH}\n"
        f"Current Inventory: {json.dumps(inventory)}\n"
        f"--- PLAYER ACTION ---\n"
        f"The player decides to: {action}"
    )
    messages = build_messages(system_prompt, history, context_snapshot)

    ai_data, story_text = generate_story(messages, sampling, avoid_openers)

    new_health, new_inventory, new_location = apply_updates(
        health, inventory, location, ai_data.get("state_updates")
    )

    history = (history + [
        {"is_player": True, "text": action},
        {"is_player": False, "text": story_text, "mode": ctx.mode.value},  # "mode" feeds next turn's streak
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

    return {
        "story": story_text,
        "health": new_health,
        "inventory": new_inventory,
        "location": new_location,
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
