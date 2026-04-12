"""
inference.py — YouTube Addiction Controller
LLM agent interacts with the OpenEnv environment.
Emits structured [START], [STEP], [END] logs as required by the hackathon.
Uses raw HTTP requests to avoid OpenAI SDK proxy conflicts in Scaler's sandbox.
Includes full try/except guards so the script NEVER exits with a non-zero code.
"""

import os
import json
import time
import requests

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

FALLBACK_ENV_URL = "https://team-youtube-ctrl-youtube-addiction-controller.hf.space"
FALLBACK_API_URL = "https://api.openai.com/v1"
VALID_ACTIONS = ("allow", "block", "suggest_break")


def rule_based_action(observation: dict) -> str:
    """
    Deterministic fallback policy — used when LLM call fails or no API key is set.
    Mirrors the reward logic so it still earns reasonable scores.
    """
    youtube_minutes = observation.get("youtube_minutes", 0)
    productive_minutes = observation.get("productive_minutes", 0)
    has_deadline = observation.get("has_deadline", False)
    videos_watched = observation.get("videos_watched", 0)
    user_mood = observation.get("user_mood", "")

    # Worked 25+ min straight → suggest break
    if productive_minutes >= 25 and youtube_minutes == 0:
        return "suggest_break"

    # Deadline + on YouTube → block
    if has_deadline and youtube_minutes > 0:
        return "block"

    # Binge-watching 3+ videos → block
    if videos_watched >= 3:
        return "block"

    # YouTube > 50% of total session → block
    total = youtube_minutes + productive_minutes
    if total > 0 and youtube_minutes / total > 0.5:
        return "block"

    # Tired mood after long session → suggest break
    if user_mood in ("tired", "exhausted") and total > 30:
        return "suggest_break"

    return "allow"


def call_llm(observation: dict) -> str:
    """
    Call LLM via raw HTTP — avoids OpenAI SDK proxy conflicts.
    Falls back to rule_based_action if anything goes wrong.
    """
    api_key = (
        os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("HF_TOKEN")
        or ""
    )

    # If no API key at all, skip network call entirely
    if not api_key:
        print("[INFO] No API key found — using rule-based fallback policy.", flush=True)
        return rule_based_action(observation)

    base_url = (os.environ.get("API_BASE_URL") or FALLBACK_API_URL).rstrip("/")
    model = os.environ.get("MODEL_NAME") or "gpt-4o-mini"

    obs_str = json.dumps(observation, indent=2)
    user_msg = (
        f"Current session state:\n{obs_str}\n\n"
        "What action should you take? Reply with ONLY: allow, block, or suggest_break"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 10,
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        action = data["choices"][0]["message"]["content"].strip().lower()

        # Strip punctuation / extra whitespace
        action = action.strip(".,!? \n")

        if action not in VALID_ACTIONS:
            print(f"[WARN] LLM returned unexpected action '{action}' — using fallback.", flush=True)
            return rule_based_action(observation)

        return action

    except requests.exceptions.Timeout:
        print("[WARN] LLM request timed out — using rule-based fallback.", flush=True)
        return rule_based_action(observation)

    except requests.exceptions.HTTPError as e:
        print(f"[WARN] LLM HTTP error: {e} — using rule-based fallback.", flush=True)
        return rule_based_action(observation)

    except Exception as e:
        print(f"[WARN] LLM call failed: {e} — using rule-based fallback.", flush=True)
        return rule_based_action(observation)


def call_env(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    """Call the OpenEnv environment server."""
    env_url = (os.environ.get("ENV_URL") or FALLBACK_ENV_URL).rstrip("/")
    url = f"{env_url}{endpoint}"

    try:
        if method == "POST":
            resp = requests.post(url, json=payload, timeout=30)
        else:
            resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.Timeout:
        raise RuntimeError(f"Environment request to {url} timed out.")

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Environment HTTP error at {url}: {e}")

    except Exception as e:
        raise RuntimeError(f"Environment call failed at {url}: {e}")


def run_episode(task_id: str) -> dict:
    """Run a full episode for a single task."""
    try:
        reset_result = call_env("/reset", method="POST", payload={"task_id": task_id})
    except RuntimeError as e:
        print(f"[WARN] Could not reset task {task_id}: {e} — skipping.", flush=True)
        return {"task_id": task_id, "score": 0.0, "steps": 0, "total_reward": 0.0}

    observation = reset_result.get("observation", {})

    step_num = 0
    total_reward = 0.0
    max_steps = 50  # Safety cap to prevent infinite loops

    while not observation.get("done", False) and step_num < max_steps:
        step_num += 1

        try:
            action = call_llm(observation)
        except Exception as e:
            print(f"[WARN] Action selection failed at step {step_num}: {e} — defaulting to allow.", flush=True)
            action = "allow"

        try:
            step_result = call_env("/step", method="POST", payload={"action": action})
        except RuntimeError as e:
            print(f"[WARN] Step call failed at step {step_num}: {e} — ending episode.", flush=True)
            break

        reward = step_result.get("reward", 0.0)
        total_reward += reward
        observation = step_result.get("observation", {})

        print(
            f"[STEP] task={task_id} step={step_num} action={action} reward={reward} "
            f"done={observation.get('done', False)} "
            f"youtube_minutes={observation.get('youtube_minutes', 0)} "
            f"productive_minutes={observation.get('productive_minutes', 0)} "
            f"user_mood={observation.get('user_mood', '')}",
            flush=True,
        )

        time.sleep(0.1)

    score = round(min(1.0, total_reward / max(step_num, 1)), 4)
    return {
        "task_id": task_id,
        "score": score,
        "steps": step_num,
        "total_reward": round(total_reward, 4),
    }


def main():
    model = os.environ.get("MODEL_NAME") or "gpt-4o-mini"
    env_url = (os.environ.get("ENV_URL") or FALLBACK_ENV_URL).rstrip("/")

    print(
        f"[START] task=youtube_addiction_controller model={model} env_url={env_url}",
        flush=True,
    )

    all_results = []
    for task_id in TASKS:
        print(f"[STEP] event=task_start task={task_id}", flush=True)

        try:
            result = run_episode(task_id)
        except Exception as e:
            print(f"[WARN] Unhandled error in task {task_id}: {e} — recording zero score.", flush=True)
            result = {"task_id": task_id, "score": 0.0, "steps": 0, "total_reward": 0.0}

        all_results.append(result)
        print(
            f"[STEP] event=task_end task={result['task_id']} "
            f"score={result['score']} steps={result['steps']}",
            flush=True,
        )

    overall_score = round(sum(r["score"] for r in all_results) / len(all_results), 4)
    total_steps = sum(r["steps"] for r in all_results)

    print(
        f"[END] task=youtube_addiction_controller score={overall_score} steps={total_steps}",
        flush=True,
    )


if __name__ == "__main__":
    main()