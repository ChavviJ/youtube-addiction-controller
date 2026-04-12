"""
inference.py — YouTube Addiction Controller
LLM agent (via OpenAI client) interacts with the OpenEnv environment.
Emits structured [START], [STEP], [END] logs as required by the hackathon.
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


def get_client():
    """Create OpenAI client using the validator-injected API_KEY and API_BASE_URL."""
    from openai import OpenAI
    # Scaler injects API_KEY and API_BASE_URL — must use these exact names
    api_key = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", ""))
    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise ValueError("API_KEY environment variable is not set")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_env(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    url = f"{env_url}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=payload, timeout=30)
        else:
            resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[STEP] event=env_error endpoint={endpoint} error={str(e)}", flush=True)
        raise


def get_action_from_llm(observation: dict) -> str:
    obs_str = json.dumps(observation, indent=2)
    user_msg = f"""Current session state:
{obs_str}

What action should you take? Reply with ONLY: allow, block, or suggest_break"""

    try:
        client = get_client()
        model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=10,
            temperature=0.2,
        )
        action = response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"[STEP] event=llm_error error={str(e)}", flush=True)
        action = "allow"

    if action not in ("allow", "block", "suggest_break"):
        action = "allow"
    return action


def run_episode(task_id: str) -> dict:
    reset_result = call_env("/reset", method="POST", payload={"task_id": task_id})
    observation = reset_result["observation"]

    step_num = 0
    total_reward = 0.0

    while not observation.get("done", False):
        step_num += 1
        action = get_action_from_llm(observation)
        step_result = call_env("/step", method="POST", payload={"action": action})

        reward = step_result["reward"]
        total_reward += reward
        observation = step_result["observation"]

        print(
            f"[STEP] task={task_id} step={step_num} action={action} reward={reward} "
            f"done={observation.get('done', False)} "
            f"youtube_minutes={observation.get('youtube_minutes', 0)} "
            f"productive_minutes={observation.get('productive_minutes', 0)} "
            f"user_mood={observation.get('user_mood', '')}",
            flush=True
        )

        time.sleep(0.1)

    score = round(min(1.0, total_reward / max(step_num, 1)), 4)
    return {"task_id": task_id, "score": score, "steps": step_num, "total_reward": round(total_reward, 4)}


def main():
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")

    print(f"[START] task=youtube_addiction_controller model={model} env_url={env_url}", flush=True)

    all_results = []

    for task_id in TASKS:
        print(f"[STEP] event=task_start task={task_id}", flush=True)
        result = run_episode(task_id)
        all_results.append(result)
        print(
            f"[STEP] event=task_end task={result['task_id']} score={result['score']} steps={result['steps']}",
            flush=True
        )

    overall_score = round(sum(r["score"] for r in all_results) / len(all_results), 4)

    print(f"[END] task=youtube_addiction_controller score={overall_score} steps={sum(r['steps'] for r in all_results)}", flush=True)


if __name__ == "__main__":
    main()