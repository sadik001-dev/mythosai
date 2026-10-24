import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. Initialize environment properties
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, OPENROUTER_API_KEY]):
    raise ValueError("CRITICAL ERROR: Make sure SUPABASE_URL, SUPABASE_KEY, and OPENROUTER_API_KEY are configured in your .env file!")

# 2. Setup Cloud Connections
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Visual payloads mapping incoming frontend data packets
class StartGameRequest(BaseModel):
    universe_name: str
    genre: str

class GameTurnRequest(BaseModel):
    session_id: str
    player_action: str

def get_system_prompt(genre: str) -> str:
    """Strict structural orchestration prompt for formatting open-source outputs."""
    return f"""You are the advanced Game Engine and Dungeon Master for MythosAI.
The current universe genre is: {genre}.

Your duties:
1. Maintain a gritty, highly immersive narrative style aligned with the genre.
2. Never break character or explain that you are an AI.
3. You MUST respond ONLY in a valid JSON format containing exactly two keys: "story_text" and "state_updates".
4. "story_text" must contain exactly one or two short paragraphs describing the consequence of the player's action and presenting a compelling dilemma.
5. "state_updates" must strictly track adjustments to stats based on the narrative outcome (e.g., "health_change": -10, "add_inventory": ["Plasma Pistol"], "remove_inventory": ["Rusty Dagger"], "new_location": "Neon Alleyway"). Do not include changes if nothing happens.

Example Output Structure:
{{
    "story_text": "The iron hinges scream as you kick the locker panel inward... [Dilemma text]",
    "state_updates": {{
        "health_change": 0,
        "add_inventory": ["Lead Pipe"],
        "new_location": "Silo Sector B"
    }}
}}"""

@app.post("/api/start-game")
async def start_game(request: StartGameRequest):
    """Initializes a new game session directly inside your Supabase Cloud tables."""
    # Define generic start allocations depending on selected module framework
    default_items = ["Ration Cube", "Flashlight"]
    if "Cyberpunk" in request.genre:
        default_items = ["Data-Deck", "50 Credits"]
    elif "Fantasy" in request.genre:
        default_items = ["Iron Sword", "Elixir of Life"]

    try:
        data = supabase.table("game_states").insert({
            "universe_name": request.universe_name,
            "genre": request.genre,
            "health": 100,
            "inventory": default_items,
            "current_location": "Starting Zone",
            "story_history": []
        }).execute()
        
        new_session = data.data[0]
        return {
            "session_id": new_session["id"],
            "health": new_session["health"],
            "inventory": new_session["inventory"],
            "location": new_session["current_location"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Initialization Failed: {str(e)}")

@app.post("/api/game-turn")
async def game_turn(request: GameTurnRequest):
    """Core runtime engine: Coordinates state data history records, triggers AI models, maps updates back to Supabase."""
    try:
        # Fetch current record context from Supabase cloud database
        state_query = supabase.table("game_states").select("*").eq("id", request.session_id).single().execute()
        current_state = state_query.data
        if not current_state:
            raise HTTPException(status_code=404, detail="Game session not found.")
        
        # Unpack operational values
        genre = current_state["genre"]
        health = current_state["health"]
        inventory = current_state["inventory"] or []
        location = current_state["current_location"]
        history = current_state["story_history"] or []

        # Package current running metrics as a prompt snapshot framework context
        context_snapshot = (
            f"--- CURRENT STATE ---\n"
            f"Location: {location}\n"
            f"Current Health: {health}/100\n"
            f"Current Inventory: {json.dumps(inventory)}\n"
            f"--- PLAYER ACTION ---\n"
            f"The player decides to: {request.player_action}"
        )

        messages = [{"role": "system", "content": get_system_prompt(genre)}]
        # Append historical contextual updates to ensure system path memory optimization
        for turn in history[-4:]:
            messages.append({"role": "user" if turn["is_player"] else "assistant", "content": turn["text"]})
        messages.append({"role": "user", "content": context_snapshot})

        # Submit data package directly across OpenRouter gateway pipeline interface
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-3-8b-instruct:free",
            "messages": messages,
            "response_format": {"type": "json_object"}
        }

        response = requests.post("https://openrouter.ai", json=payload, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"OpenRouter Gateway Failure: {response.text}")

        # Parse structural JSON engine constraints returns
        ai_raw = response.json()['choices'][0]['message']['content']
        ai_data = json.loads(ai_raw)
        
        story_text = ai_data.get("story_text", "")
        updates = ai_data.get("state_updates", {})

        # Compute calculated stat updates adjustments safely
        new_health = max(0, min(100, health + updates.get("health_change", 0)))
        new_location = updates.get("new_location", location)
        
        updated_inventory = list(inventory)
        for item in updates.get("add_inventory", []):
            if item not in updated_inventory:
                updated_inventory.append(item)
        for item in updates.get("remove_inventory", []):
            if item in updated_inventory:
                updated_inventory.remove(item)

        # Append current action variables entries to tracking logs
        history.append({"is_player": True, "text": request.player_action})
        history.append({"is_player": False, "text": story_text})

        # Save structured changes cleanly back into Supabase cloud table database infrastructure records
        supabase.table("game_states").update({
            "health": new_health,
            "inventory": updated_inventory,
            "current_location": new_location,
            "story_history": history
        }).eq("id", request.session_id).execute()

        return {
            "story": story_text,
            "health": new_health,
            "inventory": updated_inventory,
            "location": new_location
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Game Turn Processing Failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
