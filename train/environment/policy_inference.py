from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from openenv_service import PandemicPolicyAction, PandemicPolicyObservation, ROOT_DIR

LOOKBACK = 14

NPI_LAYOUT: list[tuple[str, int]] = [
    ("c1m_school_closing", 3),
    ("c2m_workplace_closing", 3),
    ("c3m_cancel_public_events", 2),
    ("c4m_restrictions_on_gatherings", 4),
    ("c5m_close_public_transport", 2),
    ("c6m_stay_at_home_requirements", 3),
    ("c7m_restrictions_on_internal_movement", 2),
    ("c8ev_international_travel_controls", 4),
    ("h1_public_information_campaigns", 2),
    ("h2_testing_policy", 3),
    ("h3_contact_tracing", 2),
    ("h6m_facial_coverings", 4),
    ("h7_vaccination_policy", 5),
    ("e1_income_support", 2),
    ("e2_debt_contract_relief", 1),
]
NPI_KEYS = [name for name, _ in NPI_LAYOUT]
NPI_DIMS = [max_level + 1 for _, max_level in NPI_LAYOUT]

OPENENV_ACTION_TO_POLICY = {
    "school_closing": "c1m_school_closing",
    "workplace_closing": "c2m_workplace_closing",
    "cancel_public_events": "c3m_cancel_public_events",
    "restrictions_on_gatherings": "c4m_restrictions_on_gatherings",
    "close_public_transport": "c5m_close_public_transport",
    "stay_at_home": "c6m_stay_at_home_requirements",
    "internal_movement": "c7m_restrictions_on_internal_movement",
    "international_travel": "c8ev_international_travel_controls",
    "facial_coverings": "h6m_facial_coverings",
    "vaccination_policy": "h7_vaccination_policy",
    "income_support": "e1_income_support",
    "debt_relief": "e2_debt_contract_relief",
}


class RLPolicyController:
    def __init__(
        self,
        model_path: Path,
        vecnormalize_path: Path | None,
        lookback: int = LOOKBACK,
        deterministic: bool = True,
    ):
        try:
            import gymnasium as gym
            from gymnasium import spaces
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "RL dependencies missing (requires gymnasium, stable-baselines3, torch)."
            ) from exc

        self.model_path = model_path
        self.lookback = lookback
        self.deterministic = deterministic
        self.history: deque[np.ndarray] = deque(maxlen=lookback)

        self.model = PPO.load(str(model_path), device="cpu")

        self.vec_norm: Any | None = None
        if vecnormalize_path is not None and vecnormalize_path.exists():

            class _InferenceObsEnv(gym.Env):
                metadata = {"render_modes": []}

                def __init__(self):
                    super().__init__()
                    self.observation_space = spaces.Dict(
                        {
                            "time_series": spaces.Box(
                                low=-np.inf,
                                high=np.inf,
                                shape=(lookback, 7),
                                dtype=np.float32,
                            ),
                            "static": spaces.Box(
                                low=-np.inf,
                                high=np.inf,
                                shape=(5,),
                                dtype=np.float32,
                            ),
                        }
                    )
                    self.action_space = spaces.MultiDiscrete(NPI_DIMS)

                def _zero_obs(self) -> dict[str, np.ndarray]:
                    return {
                        "time_series": np.zeros((lookback, 7), dtype=np.float32),
                        "static": np.zeros((5,), dtype=np.float32),
                    }

                def reset(self, seed=None, options=None):
                    super().reset(seed=seed)
                    return self._zero_obs(), {}

                def step(self, action):
                    return self._zero_obs(), 0.0, False, False, {}

            dummy = DummyVecEnv([_InferenceObsEnv])
            self.vec_norm = VecNormalize.load(str(vecnormalize_path), dummy)
            self.vec_norm.training = False
            self.vec_norm.norm_reward = False

    @staticmethod
    def default_model_candidates() -> list[Path]:
        model_dir = ROOT_DIR / "train" / "environment" / "models"
        return [
            model_dir / "best_model.zip",
            model_dir / "ppo_epidemic_us_final.zip",
            model_dir / "ppo_epidemic_us_400000_steps.zip",
            model_dir / "ppo_epidemic_us_200000_steps.zip",
        ]

    @classmethod
    def from_environment(cls) -> "RLPolicyController":
        model_override = os.getenv("RL_MODEL_PATH")
        vecnorm_override = os.getenv("RL_VECNORMALIZE_PATH")
        deterministic = os.getenv("RL_DETERMINISTIC", "1") != "0"

        model_path: Path | None = None
        if model_override:
            candidate = Path(model_override)
            if not candidate.is_absolute():
                candidate = ROOT_DIR / candidate
            if candidate.exists():
                model_path = candidate
            elif candidate.suffix != ".zip" and candidate.with_suffix(".zip").exists():
                model_path = candidate.with_suffix(".zip")
            else:
                raise FileNotFoundError(f"RL model not found: {candidate}")
        else:
            for candidate in cls.default_model_candidates():
                if candidate.exists():
                    model_path = candidate
                    break

        if model_path is None:
            raise FileNotFoundError(
                "No PPO model checkpoint found in train/environment/models"
            )

        if vecnorm_override:
            vec_path = Path(vecnorm_override)
            if not vec_path.is_absolute():
                vec_path = ROOT_DIR / vec_path
        else:
            vec_path = model_path.parent / "vec_normalize_us.pkl"

        return cls(
            model_path=model_path,
            vecnormalize_path=vec_path if vec_path.exists() else None,
            lookback=LOOKBACK,
            deterministic=deterministic,
        )

    @staticmethod
    def _ts_row(obs: PandemicPolicyObservation) -> np.ndarray:
        return np.array(
            [
                obs.stringency_index / 100.0,
                obs.gdp_index / 100.0,
                obs.re,
                obs.new_cases_norm,
                0.0,
                obs.weekly_growth_rate,
                0.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _static_row(obs: PandemicPolicyObservation) -> np.ndarray:
        return np.array(
            [
                obs.susceptible,
                obs.infected,
                obs.recovered,
                obs.vaccinated_pct / 100.0,
                obs.fully_vaccinated_pct / 100.0,
            ],
            dtype=np.float32,
        )

    def reset_history(self, obs: PandemicPolicyObservation) -> None:
        self.history.clear()
        row = self._ts_row(obs)
        for _ in range(self.lookback):
            self.history.append(row.copy())

    def _obs_for_model(self, obs: PandemicPolicyObservation) -> dict[str, np.ndarray]:
        row = self._ts_row(obs)
        if not self.history:
            self.reset_history(obs)
        else:
            self.history.append(row)

        ts = np.stack(list(self.history), axis=0).astype(np.float32)
        static = self._static_row(obs)

        model_obs: dict[str, np.ndarray] = {
            "time_series": ts[np.newaxis, ...],
            "static": static[np.newaxis, ...],
        }
        if self.vec_norm is not None:
            model_obs = self.vec_norm.normalize_obs(model_obs)
        return model_obs

    def predict_action(self, obs: PandemicPolicyObservation) -> PandemicPolicyAction:
        model_obs = self._obs_for_model(obs)
        action_raw, _ = self.model.predict(model_obs, deterministic=self.deterministic)
        action_arr = np.asarray(action_raw)
        if action_arr.ndim > 1:
            action_arr = action_arr[0]

        npi_levels: dict[str, int] = {}
        for idx, npi_key in enumerate(NPI_KEYS):
            max_level = int(NPI_DIMS[idx] - 1)
            value = int(np.clip(int(action_arr[idx]), 0, max_level))
            npi_levels[npi_key] = value

        action_payload = {
            action_key: npi_levels[policy_key]
            for action_key, policy_key in OPENENV_ACTION_TO_POLICY.items()
        }
        return PandemicPolicyAction(**action_payload)
