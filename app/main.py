"""
YouTube Addiction Controller — FastAPI Server
Exposes OpenEnv-compliant endpoints: /reset, /step, /state, /grade
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from app.environment import YouTubeAddictionEnv
from app.graders import run_all_graders

app = FastAPI(
    title="YouTube Addiction Controller",
    description="OpenEnv-compliant RL environment for screen-time management",
    version="1.0.0",
)

# Global environment instance (single session for hackathon purposes)
env = YouTubeAddictionEnv()


# ── Request/Response Models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_casual"


class StepRequest(BaseModel):
    action: str  # "allow" | "block" | "suggest_break"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "YouTube Addiction Controller",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/grade", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(request: ResetRequest = None):
    task_id = request.task_id if request else "task_casual"
    if task_id not in ("task_casual", "task_addict", "task_binge_procrastinator"):
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    observation = env.reset(task_id=task_id)
    return {"observation": observation, "task_id": task_id}


@app.post("/step")
def step(request: StepRequest):
    if request.action not in ("allow", "block", "suggest_break"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{request.action}'. Must be: allow | block | suggest_break"
        )
    result = env.step(request.action)
    return result


@app.get("/state")
def state():
    return env.state_dict()


@app.get("/grade")
def grade():
    """Run all graders and return scores for all 3 tasks."""
    results = run_all_graders()
    return results


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "task_casual", "name": "Casual User", "difficulty": "easy"},
            {"id": "task_addict", "name": "Addicted User", "difficulty": "medium"},
            {"id": "task_binge_procrastinator", "name": "Binge Procrastinator", "difficulty": "hard"},
        ]
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)
