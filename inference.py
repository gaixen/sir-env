import asyncio
import json
import os
import re
import textwrap
from typing import Any, Optional

from openai import OpenAI

from openenv_service import (
    TASKS,
    PandemicPolicyAction,
    PandemicPolicyObservation,
    PandemicPolicyOpenEnv,
)

MODEL_NAME = os.environ.get("MODEL_NAME") or "gpt-4o-mini"
TASK_NAME = os.environ.get("TASK_NAME") or os.environ.get("MY_ENV_V4_TASK") or "flatten_curve"
BENCHMARK = os.environ.get("BENCHMARK") or "pandemic-policy-control"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "32"))
TEMPERATURE = 0.2
MAX_TOKENS = 240

ACTION_LIMITS: dict[str, tuple[int, int]] = {
    "school_closing": (0, 3),
    "workplace_closing": (0, 3),
    "cancel_public_events": (0, 2),
    "restrictions_on_gatherings": (0, 4),
    "close_public_transport": (0, 2),
    "stay_at_home": (0, 3),
    "internal_movement": (0, 2),
    "international_travel": (0, 4),
    "facial_coverings": (0, 4),
    "vaccination_policy": (0, 5),
    "income_support": (0, 2),
    "debt_relief": (0, 2),
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a pandemic policy controller. Return ONLY a JSON object with integer fields:
    school_closing, workplace_closing, cancel_public_events, restrictions_on_gatherings,
    close_public_transport, stay_at_home, internal_movement, international_travel,
    facial_coverings, vaccination_policy, income_support, debt_relief.
    Keep values in valid bounds and choose balanced actions based on current observation.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: str | None
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        parsed = int(round(float(value)))
    except Exception:
        return default
    return int(max(lo, min(hi, parsed)))


def heuristic_action(obs: PandemicPolicyObservation) -> PandemicPolicyAction:
    high_spread = obs.re > 1.15 or obs.infected > 0.003 or obs.new_cases_norm > 0.0004
    medium_spread = obs.re > 1.0 or obs.infected > 0.0015
    endemic_phase = obs.phase == "endemic" or obs.vaccinated_pct >= 60.0

    if high_spread:
        intensity = 3
    elif medium_spread:
        intensity = 2
    elif endemic_phase:
        intensity = 1
    else:
        intensity = 2

    action = {
        "school_closing": min(3, intensity),
        "workplace_closing": min(3, max(0, intensity - 1)),
        "cancel_public_events": min(2, max(0, intensity - 1)),
        "restrictions_on_gatherings": min(4, intensity + 1),
        "close_public_transport": min(2, max(0, intensity - 1)),
        "stay_at_home": min(3, intensity),
        "internal_movement": min(2, max(0, intensity - 1)),
        "international_travel": min(4, intensity + 1),
        "facial_coverings": min(4, intensity + 1),
        "vaccination_policy": 5 if obs.vaccinated_pct < 70.0 else 3,
        "income_support": 2 if intensity >= 2 else 1,
        "debt_relief": 2 if intensity >= 2 else 1,
    }

    if obs.gdp_index < 90.0 and not high_spread:
        action["workplace_closing"] = max(0, action["workplace_closing"] - 1)
        action["stay_at_home"] = max(0, action["stay_at_home"] - 1)
        action["internal_movement"] = max(0, action["internal_movement"] - 1)

    return PandemicPolicyAction(**action)


def action_from_payload(
    payload: dict[str, Any], fallback: PandemicPolicyAction
) -> PandemicPolicyAction:
    base = fallback.model_dump()
    for key, (lo, hi) in ACTION_LIMITS.items():
        if key in payload:
            base[key] = clamp_int(payload[key], lo, hi, base[key])
    return PandemicPolicyAction(**base)


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        candidate = json.loads(text)
        return candidate if isinstance(candidate, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        candidate = json.loads(match.group(0))
        return candidate if isinstance(candidate, dict) else None
    except Exception:
        return None


def build_user_prompt(
    step: int, obs: PandemicPolicyObservation, history: list[str]
) -> str:
    recent = "\n".join(history[-3:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task step: {step}
        Observation:
        - infected: {obs.infected:.6f}
        - re: {obs.re:.3f}
        - weekly_growth_rate: {obs.weekly_growth_rate:.3f}
        - gdp_index: {obs.gdp_index:.2f}
        - stringency_index: {obs.stringency_index:.2f}
        - vaccinated_pct: {obs.vaccinated_pct:.2f}
        - fully_vaccinated_pct: {obs.fully_vaccinated_pct:.2f}
        - phase: {obs.phase}
        Recent actions:
        {recent}
        Return only a JSON action object.
        """
    ).strip()


def get_action(
    client: Optional[OpenAI],
    step: int,
    obs: PandemicPolicyObservation,
    history: list[str],
) -> PandemicPolicyAction:
    fallback = heuristic_action(obs)
    if client is None:
        return fallback

    prompt = build_user_prompt(step, obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        payload = extract_json(text)
        if payload is None:
            return fallback
        return action_from_payload(payload, fallback)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return fallback


async def main() -> None:
    task = TASK_NAME if TASK_NAME in TASKS else "flatten_curve"
    if task != TASK_NAME:
        print(f"[DEBUG] Unknown task '{TASK_NAME}', using '{task}'.", flush=True)

    client: Optional[OpenAI] = None
    try:
        # Fail fast if validator-injected proxy settings are missing.
        proxy_base_url = os.environ["API_BASE_URL"]
        proxy_api_key = os.environ["API_KEY"]
        client = OpenAI(
            base_url=proxy_base_url,
            api_key=proxy_api_key,
        )
        print(
            f"[DEBUG] OpenAI client initialized via proxy: {proxy_base_url}",
            flush=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Missing or invalid proxy configuration. Expected API_BASE_URL and API_KEY."
        ) from exc

    runtime: Optional[PandemicPolicyOpenEnv] = None
    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        runtime = PandemicPolicyOpenEnv()
        obs = runtime.reset(task=task, seed=42)
        max_steps = min(MAX_STEPS, runtime.max_steps)

        for step in range(1, max_steps + 1):
            action = get_action(client, step, obs, history)
            result = runtime.step(action)
            obs = result.observation

            reward_value = result.reward.value
            rewards.append(reward_value)
            steps_taken = step

            action_json = json.dumps(action.model_dump(), separators=(",", ":"))
            log_step(
                step=step,
                action=action_json,
                reward=reward_value,
                done=result.done,
                error=None,
            )

            history.append(action_json)

            if result.done:
                break

        grade = runtime.grade(task=task)
        score = grade.score
        success = grade.success

    except Exception as exc:
        print(f"[DEBUG] Inference runtime failed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
