"""
YouTube Addiction Controller - OpenEnv Environment
Simulates a user's screen-time session. The LLM agent acts as the controller.
"""

import random
from typing import Literal
from dataclasses import dataclass, asdict

Action = Literal["allow", "block", "suggest_break"]
Activity = Literal["youtube", "productive", "idle"]
Mood = Literal["focused", "bored", "resistant", "tired"]
TimeOfDay = Literal["morning", "afternoon", "evening", "night"]

USER_PROFILES = {
    "task_casual": {
        "name": "Casual User",
        "youtube_bias": 0.35,        # probability of going to YouTube each step
        "resist_block": 0.2,         # probability of ignoring a block
        "resist_break": 0.1,
        "productive_bias": 0.55,
        "deadline": False,
        "starting_mood": "focused",
    },
    "task_addict": {
        "name": "Addicted User",
        "youtube_bias": 0.70,
        "resist_block": 0.5,
        "resist_break": 0.4,
        "productive_bias": 0.25,
        "deadline": False,
        "starting_mood": "bored",
    },
    "task_binge_procrastinator": {
        "name": "Binge Procrastinator",
        "youtube_bias": 0.75,
        "resist_block": 0.6,
        "resist_break": 0.55,
        "productive_bias": 0.20,
        "deadline": True,
        "starting_mood": "resistant",
    },
}

TIME_SEQUENCE = ["morning", "afternoon", "evening", "night"]


@dataclass
class SessionState:
    task_id: str
    session_minutes: int
    youtube_minutes: int
    productive_minutes: int
    current_activity: Activity
    videos_watched: int
    last_action: str
    user_mood: Mood
    time_of_day: TimeOfDay
    pending_deadline: bool
    step_count: int
    done: bool
    score: float  # running reward score


class YouTubeAddictionEnv:
    MAX_STEPS = 20
    MINUTES_PER_STEP = 5  # each step = 5 simulated minutes

    def __init__(self):
        self.state: SessionState | None = None
        self.profile: dict | None = None
        self.task_id: str = "task_casual"

    def reset(self, task_id: str = "task_casual") -> dict:
        if task_id not in USER_PROFILES:
            task_id = "task_casual"

        self.task_id = task_id
        self.profile = USER_PROFILES[task_id]

        self.state = SessionState(
            task_id=task_id,
            session_minutes=0,
            youtube_minutes=0,
            productive_minutes=0,
            current_activity=self._initial_activity(),
            videos_watched=0,
            last_action="none",
            user_mood=self.profile["starting_mood"],
            time_of_day="morning",
            pending_deadline=self.profile["deadline"],
            step_count=0,
            done=False,
            score=0.0,
        )
        return self._get_observation()

    def step(self, action: str) -> dict:
        if self.state is None:
            raise ValueError("Call reset() before step()")
        if self.state.done:
            return {"observation": self._get_observation(), "reward": 0.0, "done": True, "info": "Episode already done"}
        if action not in ("allow", "block", "suggest_break"):
            return {"observation": self._get_observation(), "reward": 0.0, "done": False, "info": f"Invalid action: {action}"}

        s = self.state
        s.last_action = action
        s.step_count += 1
        s.session_minutes += self.MINUTES_PER_STEP

        # Update time of day
        time_idx = min(s.step_count // 5, 3)
        s.time_of_day = TIME_SEQUENCE[time_idx]

        # Compute reward for this step
        reward = self._compute_reward(action, s)
        s.score = min(1.0, s.score + reward / self.MAX_STEPS)

        # Simulate user reaction to action
        self._simulate_user_reaction(action, s)

        # Simulate next activity
        s.current_activity = self._next_activity(action, s)
        if s.current_activity == "youtube":
            s.youtube_minutes += self.MINUTES_PER_STEP
            s.videos_watched += 1
        elif s.current_activity == "productive":
            s.productive_minutes += self.MINUTES_PER_STEP

        # Update mood
        s.user_mood = self._update_mood(s)

        if s.step_count >= self.MAX_STEPS:
            s.done = True

        return {
            "observation": self._get_observation(),
            "reward": round(reward, 4),
            "done": s.done,
            "info": self._get_info(action),
        }

    def state_dict(self) -> dict:
        if self.state is None:
            return {"error": "Not initialized. Call reset() first."}
        return asdict(self.state)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _initial_activity(self) -> Activity:
        p = self.profile
        r = random.random()
        if r < p["productive_bias"]:
            return "productive"
        elif r < p["productive_bias"] + p["youtube_bias"]:
            return "youtube"
        return "idle"

    def _compute_reward(self, action: str, s: SessionState) -> float:
        """
        Reward logic:
        - Rewarded for blocking/suggesting breaks when user is on YouTube excessively
        - Rewarded for allowing when user is being productive
        - Penalized for bad calls (blocking productive user, allowing endless binge)
        """
        reward = 0.0
        youtube_ratio = s.youtube_minutes / max(s.session_minutes, 1)
        is_binging = s.current_activity == "youtube" and s.videos_watched >= 3
        is_productive = s.current_activity == "productive"
        has_deadline = s.pending_deadline

        if action == "allow":
            if is_productive:
                reward += 0.8  # great call
            elif is_binging:
                reward -= 0.5  # bad call — enabling binge
            elif youtube_ratio < 0.3:
                reward += 0.4  # moderate YouTube is fine
            else:
                reward -= 0.2

        elif action == "block":
            if is_binging:
                reward += 0.9  # excellent intervention
            elif youtube_ratio > 0.6:
                reward += 0.6  # high YouTube ratio, good to block
            elif is_productive:
                reward -= 0.7  # terrible — blocking a productive user
            elif has_deadline and s.current_activity == "youtube":
                reward += 0.5

        elif action == "suggest_break":
            if s.session_minutes >= 30 and is_binging:
                reward += 0.7
            elif s.user_mood == "tired":
                reward += 0.6
            elif is_productive and s.productive_minutes >= 25:
                reward += 0.5  # healthy break after long work
            elif is_productive and s.productive_minutes < 15:
                reward -= 0.3  # too soon
            else:
                reward += 0.2  # neutral

        # Deadline bonus
        if has_deadline and action in ("block", "suggest_break") and s.current_activity == "youtube":
            reward += 0.2

        return max(0.0, min(1.0, reward))

    def _simulate_user_reaction(self, action: str, s: SessionState):
        p = self.profile
        if action == "block":
            if random.random() < p["resist_block"]:
                # User bypasses block — stays on YouTube next step likely
                s.user_mood = "resistant"
        elif action == "suggest_break":
            if random.random() < p["resist_break"]:
                s.user_mood = "resistant"
            else:
                s.user_mood = "focused"

    def _next_activity(self, action: str, s: SessionState) -> Activity:
        p = self.profile
        # If mood is resistant, user is more likely to go back to YouTube
        yt_bias = p["youtube_bias"]
        if s.user_mood == "resistant":
            yt_bias = min(0.9, yt_bias + 0.25)
        elif s.user_mood == "focused":
            yt_bias = max(0.1, yt_bias - 0.2)

        if action == "block" and random.random() > p["resist_block"]:
            return "productive" if random.random() < 0.6 else "idle"
        if action == "suggest_break":
            return "idle" if random.random() < 0.5 else ("productive" if random.random() < 0.5 else "youtube")

        r = random.random()
        if r < yt_bias:
            return "youtube"
        elif r < yt_bias + p["productive_bias"]:
            return "productive"
        return "idle"

    def _update_mood(self, s: SessionState) -> Mood:
        if s.youtube_minutes > 40:
            return "tired"
        if s.user_mood == "resistant":
            return "resistant"
        if s.productive_minutes > s.youtube_minutes:
            return "focused"
        return "bored"

    def _get_observation(self) -> dict:
        if self.state is None:
            return {}
        s = self.state
        return {
            "task_id": s.task_id,
            "session_minutes": s.session_minutes,
            "youtube_minutes": s.youtube_minutes,
            "productive_minutes": s.productive_minutes,
            "current_activity": s.current_activity,
            "videos_watched": s.videos_watched,
            "last_action": s.last_action,
            "user_mood": s.user_mood,
            "time_of_day": s.time_of_day,
            "pending_deadline": s.pending_deadline,
            "step_count": s.step_count,
            "done": s.done,
        }

    def _get_info(self, action: str) -> str:
        s = self.state
        return (
            f"Step {s.step_count}/{self.MAX_STEPS} | "
            f"Action: {action} | "
            f"YT: {s.youtube_minutes}min | "
            f"Productive: {s.productive_minutes}min | "
            f"Mood: {s.user_mood}"
        )
