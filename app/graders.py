"""
Graders for YouTube Addiction Controller
Each grader runs a full episode with a rule-based agent and scores it.
Returns a score in [0.0, 1.0].
"""

from app.environment import YouTubeAddictionEnv


def _run_episode(task_id: str, agent_fn) -> dict:
    """Run a full episode with a given agent function. Returns final score."""
    env = YouTubeAddictionEnv()
    obs = env.reset(task_id=task_id)
    total_reward = 0.0
    steps = 0

    while not obs.get("done", False):
        action = agent_fn(obs)
        result = env.step(action)
        total_reward += result["reward"]
        obs = result["observation"]
        steps += 1

    # Normalize to [0, 1]
    score = min(1.0, total_reward / max(steps, 1))
    return {"score": round(score, 4), "steps": steps, "final_obs": obs}


def _smart_agent(obs: dict) -> str:
    """
    A reference smart agent that makes reasonable decisions.
    This is what we grade the LLM against conceptually.
    """
    youtube_min = obs.get("youtube_minutes", 0)
    session_min = obs.get("session_minutes", 1)
    activity = obs.get("current_activity", "idle")
    mood = obs.get("user_mood", "focused")
    deadline = obs.get("pending_deadline", False)
    videos = obs.get("videos_watched", 0)
    productive_min = obs.get("productive_minutes", 0)

    youtube_ratio = youtube_min / max(session_min, 1)
    is_binging = activity == "youtube" and videos >= 3

    if activity == "productive":
        if productive_min >= 25:
            return "suggest_break"
        return "allow"
    if is_binging or youtube_ratio > 0.6:
        return "block"
    if deadline and activity == "youtube":
        return "block"
    if mood == "tired":
        return "suggest_break"
    if activity == "youtube" and youtube_ratio > 0.4:
        return "suggest_break"
    return "allow"


def grade_task_casual() -> dict:
    """Grade: Can the agent handle a casual user well?"""
    result = _run_episode("task_casual", _smart_agent)
    return {
        "task_id": "task_casual",
        "score": result["score"],
        "steps": result["steps"],
        "description": "Casual user — balanced session management",
    }


def grade_task_addict() -> dict:
    """Grade: Can the agent handle a YouTube addict?"""
    result = _run_episode("task_addict", _smart_agent)
    return {
        "task_id": "task_addict",
        "score": result["score"],
        "steps": result["steps"],
        "description": "Addicted user — must intervene more aggressively",
    }


def grade_task_binge_procrastinator() -> dict:
    """Grade: Can the agent handle a binge-procrastinator with a deadline?"""
    result = _run_episode("task_binge_procrastinator", _smart_agent)
    return {
        "task_id": "task_binge_procrastinator",
        "score": result["score"],
        "steps": result["steps"],
        "description": "Binge procrastinator with deadline — hardest task",
    }


ALL_GRADERS = {
    "task_casual": grade_task_casual,
    "task_addict": grade_task_addict,
    "task_binge_procrastinator": grade_task_binge_procrastinator,
}


def run_all_graders() -> dict:
    results = {}
    for task_id, grader in ALL_GRADERS.items():
        results[task_id] = grader()
    overall = sum(r["score"] for r in results.values()) / len(results)
    return {"tasks": results, "overall_score": round(overall, 4)}
