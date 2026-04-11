"""
inference.py — YouTube Addiction Controller
LLM agent (via OpenAI client) interacts with the OpenEnv environment.
Emits structured [START], [STEP], [END] logs as required by the hackathon.
"""

import os
import json
import time
import requests
from openai import OpenAI

# ── Config from environment variables ────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

import os
HF_TOKEN = os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")

TASKS = ["task_casual", "task_addict", "task_binge_procrastinator"]

SYSTEM_PROMPT = """You are a YouTube screen-time controller AI.
You observe a user's session and decide the best action to take.

Available actions:
- "allow"         → Let the user continue what they're doing
- "block"         → Block YouTube / stop the session  
- "suggest_break" → Suggest a short break

Your goal: maximize productivity while respecting the user's wellbeing.
Rules:
- Never block a productive user
- Intervene (block/suggest_break) when YouTube use is excessive (>50% of session)
- Always suggest breaks after long productive streaks (>25 min)
- Be more aggressive when the user has a pending deadline
- Consider the user's mood: resistant users need firm blocks, tired users need breaks

Respond with ONLY one word: allow, block, or suggest_break
"""


def call_env(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    if method == "POST":
        resp = requests.post(url, json=payload, timeout=30)
    else:
        resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_action_from_llm(observation: dict) -> str:
    obs_str = json.dumps(observation, indent=2)
    user_msg = f"""Current session state:
{obs_str}

What action should you take? Reply with ONLY: allow, block, or suggest_break"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=10,
        temperature=0.2,
    )
    action = response.choices[0].message.content.strip().lower()
    # Sanitize
    if action not in ("allow", "block", "suggest_break"):
        action = "allow"
    return action


def run_episode(task_id: str) -> dict:
    # Reset environment
    reset_result = call_env("/reset", method="POST", payload={"task_id": task_id})
    observation = reset_result["observation"]

    step_num = 0
    total_reward = 0.0
    episode_log = []

    while not observation.get("done", False):
        step_num += 1
        action = get_action_from_llm(observation)
        step_result = call_env("/step", method="POST", payload={"action": action})

        reward = step_result["reward"]
        total_reward += reward
        observation = step_result["observation"]

        step_log = {
            "step": step_num,
            "action": action,
            "reward": reward,
            "observation": observation,
            "info": step_result.get("info", ""),
        }
        episode_log.append(step_log)

        # Required [STEP] log format
        print(json.dumps({
            "type": "STEP",
            "task_id": task_id,
            "step": step_num,
            "action": action,
            "reward": reward,
            "done": observation.get("done", False),
            "youtube_minutes": observation.get("youtube_minutes", 0),
            "productive_minutes": observation.get("productive_minutes", 0),
            "user_mood": observation.get("user_mood", ""),
        }), flush=True)

        time.sleep(0.1)  # be nice to the API

    score = round(min(1.0, total_reward / max(step_num, 1)), 4)
    return {"task_id": task_id, "score": score, "steps": step_num, "total_reward": round(total_reward, 4)}


def main():
    # Required [START] log
    print(json.dumps({
        "type": "START",
        "model": MODEL_NAME,
        "tasks": TASKS,
        "env_url": ENV_URL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), flush=True)

    all_results = []

    for task_id in TASKS:
        print(json.dumps({"type": "STEP", "event": "task_start", "task_id": task_id}), flush=True)
        result = run_episode(task_id)
        all_results.append(result)
        print(json.dumps({"type": "STEP", "event": "task_end", **result}), flush=True)

    overall_score = round(sum(r["score"] for r in all_results) / len(all_results), 4)

    # Required [END] log
    print(json.dumps({
        "type": "END",
        "tasks": all_results,
        "overall_score": overall_score,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), flush=True)


if __name__ == "__main__":
    main()

