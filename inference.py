import asyncio
import json
import os

from dotenv import load_dotenv

from openenv_service import (
    TASKS,
    PandemicPolicyAction,
    PandemicPolicyObservation,
    PandemicPolicyOpenEnv,
)
from train.environment.policy_inference import RLPolicyController

load_dotenv()

TASK_NAME = os.getenv("TASK_NAME") or os.getenv("MY_ENV_V4_TASK") or "flatten_curve"
BENCHMARK = os.getenv("BENCHMARK") or "pandemic-policy-control"
MAX_STEPS = int(os.getenv("MAX_STEPS", "32"))


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


async def main() -> None:
    task = TASK_NAME if TASK_NAME in TASKS else "flatten_curve"
    if task != TASK_NAME:
        print(f"[DEBUG] Unknown task '{TASK_NAME}', using '{task}'.", flush=True)

    policy: RLPolicyController | None = None
    model_label = "heuristic-fallback"
    try:
        policy = RLPolicyController.from_environment()
        model_label = f"ppo:{policy.model_path.name}"
        print(f"[DEBUG] Loaded RL policy from {policy.model_path}", flush=True)
    except Exception as exc:
        print(
            f"[DEBUG] RL policy load failed; using heuristic policy: {exc}", flush=True
        )

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=model_label)

    try:
        runtime = PandemicPolicyOpenEnv()
        obs = runtime.reset(task=task, seed=42)
        if policy is not None:
            policy.reset_history(obs)

        max_steps = min(MAX_STEPS, runtime.max_steps)

        for step in range(1, max_steps + 1):
            if policy is not None:
                action = policy.predict_action(obs)
            else:
                action = heuristic_action(obs)

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
