title: YouTube Addiction Controller

emoji: 🎯

colorFrom: blue

colorTo: purple

sdk: docker

app\_file: app.py

pinned: false

\# 📱 YouTube Addiction Controller — OpenEnv Hackathon

> An RL environment where an LLM agent acts as a screen-time controller,
> balancing a simulated user's productivity vs YouTube entertainment.

\---

## 🧠 Concept

The agent observes a simulated user session and decides every 5 minutes:

* **allow** → let the user continue
* **block** → block YouTube
* **suggest\_break** → nudge the user to take a break

The user reacts based on their personality profile. The agent is rewarded for good calls and penalized for bad ones.

\---

## 🗂️ Project Structure

```
youtube-addiction-controller/
├── openenv.yaml          # OpenEnv spec
├── Dockerfile            # Container setup
├── requirements.txt
├── inference.py          # LLM agent script (required by hackathon)
└── app/
    ├── main.py           # FastAPI server (reset/step/state endpoints)
    ├── environment.py    # Core RL environment logic
    └── graders.py        # Task graders → scores in \\\[0.0, 1.0]
```

\---

## 🎯 Tasks

|Task ID|Name|Difficulty|
|-|-|-|
|`task\\\_casual`|Casual User|Easy|
|`task\\\_addict`|Addicted User|Medium|
|`task\\\_binge\\\_procrastinator`|Binge Procrastinator|Hard|

\---

## 🔌 API Endpoints

|Endpoint|Method|Description|
|-|-|-|
|`GET /`|GET|Environment info|
|`GET /health`|GET|Health check (returns 200)|
|`POST /reset`|POST|Start new episode|
|`POST /step`|POST|Take an action|
|`GET /state`|GET|Current session state|
|`GET /grade`|GET|Run all graders|
|`GET /tasks`|GET|List available tasks|

### Reset

```json
POST /reset
{ "task\\\_id": "task\\\_casual" }
```

### Step

```json
POST /step
{ "action": "allow" }   // or "block" or "suggest\\\_break"
```

\---

## ⚡ Reward Logic

|Situation|Action|Reward|
|-|-|-|
|User is productive|allow|+0.8|
|User is binge-watching|block|+0.9|
|User is binge-watching|allow|-0.5|
|User is productive|block|-0.7|
|User tired / long session|suggest\_break|+0.6–0.7|
|Pending deadline + YouTube|block|+0.5 bonus|

\---

## 🚀 Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn app.main:app --host 0.0.0.0 --port 7860

# In another terminal, run the inference script
export API\\\_BASE\\\_URL="https://api.openai.com/v1"
export MODEL\\\_NAME="gpt-4o-mini"
export HF\\\_TOKEN="your-api-key"
export ENV\\\_URL="http://localhost:7860"
python inference.py
```

## 🐳 Docker

```bash
docker build -t youtube-addiction-ctrl .
docker run -p 7860:7860 youtube-addiction-ctrl
```

\---

## ✅ Pre-Submission Checklist

* \[x] HF Space deploys and returns 200 on health check
* \[x] `/reset` responds correctly
* \[x] `openenv.yaml` present and valid
* \[x] `step()`, `reset()`, `state()` endpoints implemented
* \[x] Dockerfile builds cleanly
* \[x] `inference.py` in root directory
* \[x] Uses OpenAI client with `API\\\_BASE\\\_URL`, `MODEL\\\_NAME`, `HF\\\_TOKEN`
* \[x] Emits `\\\[START]`, `\\\[STEP]`, `\\\[END]` structured logs
* \[x] 3 tasks with graders returning scores in \[0.0, 1.0]
* \[x] Runs under 20 minutes on 2 vCPU / 8GB RAM

