from fastapi import FastAPI
from environment import YouTubeEnv

app = FastAPI()   # ← THIS LINE IS CRITICAL

env = YouTubeEnv()

@app.get("/reset")
def reset(level: str = "easy"):
    return env.reset(level)

@app.post("/step")
def step(action: dict):
    return env.step(action["action"])

@app.get("/state")
def state():
    return env.state()