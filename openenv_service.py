from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from scipy.integrate import odeint


_THIS_FILE = Path(__file__).resolve()
ROOT_DIR = _THIS_FILE.parent if (_THIS_FILE.parent / "data").exists() else _THIS_FILE.parents[2]
DATA_PATH = ROOT_DIR / "data" / "combined_us_data.csv"
PARAMS_PATH = ROOT_DIR / "train" / "sir_params_US.json"

POPULATION = 330_000_000.0
RECOVERY_DAYS = 14


with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    _params = json.load(f)

BETA = float(_params["beta_final"])
GAMMA = float(_params["gamma_final"])
NU_TV = np.asarray(_params["nu_tv"], dtype=float)
GDP_COEFFS = np.asarray(_params["gdp_coeffs"], dtype=float)

# Existing policy column mapping reused from the RL environment.
_NPI_MAX_MAPPINGS = {
    "c1m_school_closing": 3,
    "c2m_workplace_closing": 3,
    "c3m_cancel_public_events": 2,
    "c4m_restrictions_on_gatherings": 4,
    "c5m_close_public_transport": 2,
    "c6m_stay_at_home_requirements": 3,
    "c7m_restrictions_on_internal_movement": 2,
    "c8ev_international_travel_controls": 4,
    "h1_public_information_campaigns": 2,
    "h2_testing_policy": 3,
    "h3_contact_tracing": 2,
    "h6m_facial_coverings": 4,
    "h7_vaccination_policy": 5,
    "e1_income_support": 2,
    "e2_debt_contract_relief": 1,
}

# Action fields exposed to agents and their corresponding policy columns.
ACTION_TO_POLICY = {
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

STRINGENCY_KEYS = [k for k in _NPI_MAX_MAPPINGS if not k.startswith("e")]


@dataclass(frozen=True)
class TaskConfig:
    name: str
    max_steps: int
    start_date: str
    end_date: str
    success_threshold: float


TASKS: dict[str, TaskConfig] = {
    "flatten_curve": TaskConfig(
        name="flatten_curve",
        max_steps=120,
        start_date="2020-12-20",
        end_date="2021-06-30",
        success_threshold=0.85,
    ),
    "balanced_response": TaskConfig(
        name="balanced_response",
        max_steps=180,
        start_date="2021-01-15",
        end_date="2022-02-28",
        success_threshold=0.70,
    ),
    "optimal_transition": TaskConfig(
        name="optimal_transition",
        max_steps=240,
        start_date="2020-12-20",
        end_date="2022-10-31",
        success_threshold=0.60,
    ),
}


class PandemicPolicyAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    school_closing: int = Field(ge=0, le=3)
    workplace_closing: int = Field(ge=0, le=3)
    cancel_public_events: int = Field(ge=0, le=2)
    restrictions_on_gatherings: int = Field(ge=0, le=4)
    close_public_transport: int = Field(ge=0, le=2)
    stay_at_home: int = Field(ge=0, le=3)
    internal_movement: int = Field(ge=0, le=2)
    international_travel: int = Field(ge=0, le=4)
    facial_coverings: int = Field(ge=0, le=4)
    vaccination_policy: int = Field(ge=0, le=5)
    income_support: int = Field(ge=0, le=2)
    debt_relief: int = Field(ge=0, le=2)


class PandemicPolicyObservation(BaseModel):
    susceptible: float = Field(ge=0.0, le=1.0)
    infected: float = Field(ge=0.0, le=1.0)
    recovered: float = Field(ge=0.0, le=1.0)
    re: float = Field(ge=0.0)
    new_cases_norm: float = Field(ge=0.0)
    weekly_growth_rate: float
    gdp_index: float = Field(ge=0.0, le=100.0)
    stringency_index: float = Field(ge=0.0, le=100.0)
    vaccinated_pct: float = Field(ge=0.0, le=100.0)
    fully_vaccinated_pct: float = Field(ge=0.0, le=100.0)
    day: int = Field(ge=0)
    phase: str


class PandemicPolicyReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    health_component: float = Field(ge=0.0, le=1.0)
    economic_component: float = Field(ge=0.0, le=1.0)
    stability_component: float = Field(ge=0.0, le=1.0)
    support_component: float = Field(ge=0.0, le=1.0)
    penalty_component: float = Field(ge=0.0, le=1.0)


class StepResponse(BaseModel):
    observation: PandemicPolicyObservation
    reward: PandemicPolicyReward
    done: bool
    info: dict[str, Any]


class ResetRequest(BaseModel):
    task: str = Field(default="flatten_curve")
    seed: int | None = None


class ResetResponse(BaseModel):
    task: str
    observation: PandemicPolicyObservation
    max_steps: int


class StateResponse(BaseModel):
    task: str
    day: int
    max_steps: int
    done: bool
    cumulative_reward: float
    last_observation: PandemicPolicyObservation


class GradeRequest(BaseModel):
    task: str | None = None


class GradeResponse(BaseModel):
    task: str
    score: float = Field(ge=0.0, le=1.0)
    success: bool
    threshold: float
    breakdown: dict[str, float]


class HealthResponse(BaseModel):
    status: str


class PandemicPolicyOpenEnv:
    def __init__(self, data_path: Path = DATA_PATH):
        self.df = self._load_and_prepare_data(data_path)
        self.task_name = "flatten_curve"
        self.task_cfg = TASKS[self.task_name]
        self.episode_df = pd.DataFrame()

        self.day = 0
        self.max_steps = 0
        self.done = False
        self.cumulative_reward = 0.0

        self.S = 0.0
        self.I = 0.0
        self.R = 0.0
        self.prev_I = 0.0
        self.prev_stringency = 0.0
        self.vaccinated_pct = 0.0
        self.fully_vaccinated_pct = 0.0
        self.last_action: PandemicPolicyAction | None = None
        self.same_action_streak = 0

        self.infected_history: deque[float] = deque(maxlen=8)
        self.last_observation: PandemicPolicyObservation | None = None
        self.trajectory: list[dict[str, Any]] = []

        self.reset(task=self.task_name, seed=0)

    @staticmethod
    def _load_and_prepare_data(data_path: Path) -> pd.DataFrame:
        if not data_path.exists():
            raise FileNotFoundError(f"missing data file: {data_path}")

        df = pd.read_csv(data_path, low_memory=False)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        if "date" not in df.columns:
            raise KeyError("expected 'date' column in combined dataset")

        df["date"] = pd.to_datetime(df["date"])
        if "country" in df.columns:
            df = df[df["country"] == "United States"].copy()

        df = df.sort_values("date").reset_index(drop=True)

        for col in ["total_cases", "total_deaths", "stringency_index", "GDP_scaled"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["total_cases"] = df["total_cases"].ffill().fillna(0.0)
        df["total_deaths"] = df["total_deaths"].ffill().fillna(0.0)
        df["stringency_index"] = df["stringency_index"].ffill().fillna(0.0)
        df["GDP_scaled"] = df["GDP_scaled"].ffill().fillna(90.0)

        total_recovered = (
            df["total_cases"].shift(RECOVERY_DAYS).fillna(0.0) - df["total_deaths"]
        ).clip(lower=0.0)
        active_infected = (
            df["total_cases"] - total_recovered - df["total_deaths"]
        ).clip(lower=0.0)

        df["S"] = ((POPULATION - df["total_cases"]) / POPULATION).clip(
            lower=0.0, upper=1.0
        )
        df["I"] = (active_infected / POPULATION).clip(lower=0.0, upper=1.0)
        df["R"] = (total_recovered / POPULATION).clip(lower=0.0, upper=1.0)

        if "people_vaccinated_per_hundred" not in df.columns:
            df["people_vaccinated_per_hundred"] = 0.0
        if "people_fully_vaccinated_per_hundred" not in df.columns:
            df["people_fully_vaccinated_per_hundred"] = 0.0

        for key in _NPI_MAX_MAPPINGS:
            if key not in df.columns:
                df[key] = 0
            df[key] = pd.to_numeric(df[key], errors="coerce").fillna(0).astype(int)

        return df

    @staticmethod
    def _compute_gdp(stringency: float) -> float:
        poly = np.poly1d(GDP_COEFFS)
        return float(np.clip(poly(stringency), 0.0, 100.0))

    @staticmethod
    def _compute_re(s_norm: float, S: float) -> float:
        return float(max(0.0, BETA * (1.0 - s_norm) * S / GAMMA))

    @staticmethod
    def _ode_lockdown_nu(
        y: list[float], _t: float, beta: float, gamma: float, nu: float, s: float
    ) -> list[float]:
        Sus, Inf, Rem = y
        eff = beta * (1.0 - s)
        dS = -eff * Sus * Inf - nu * Sus
        dI = eff * Sus * Inf - gamma * Inf
        dR = gamma * Inf + nu * Sus
        return [dS, dI, dR]

    def _step_sir(self, s_norm: float, nu: float) -> tuple[float, float, float]:
        y0 = [self.S, self.I, self.R]
        t = np.linspace(0.0, 1.0, 2)
        sol = odeint(
            self._ode_lockdown_nu, y0, t, args=(BETA, GAMMA, nu, s_norm), hmax=1.0
        )
        S_new, I_new, R_new = map(float, sol[-1])
        S_new = float(np.clip(S_new, 0.0, 1.0))
        I_new = float(np.clip(I_new, 0.0, 1.0))
        R_new = float(np.clip(R_new, 0.0, 1.0))
        total = S_new + I_new + R_new
        if total > 0:
            S_new, I_new, R_new = S_new / total, I_new / total, R_new / total
        return S_new, I_new, R_new

    @staticmethod
    def _policy_to_stringency(policy: dict[str, int]) -> float:
        parts = []
        for key in STRINGENCY_KEYS:
            max_v = _NPI_MAX_MAPPINGS[key]
            val = int(np.clip(policy.get(key, 0), 0, max_v))
            parts.append((val / max_v) * 100.0 if max_v > 0 else 0.0)
        if not parts:
            return 0.0
        return float(np.mean(parts))

    def _base_policy_row(self) -> dict[str, int]:
        idx = min(self.day, len(self.episode_df) - 1)
        row = self.episode_df.iloc[idx]
        return {k: int(row.get(k, 0) or 0) for k in _NPI_MAX_MAPPINGS}

    def _action_to_policy(self, action: PandemicPolicyAction) -> dict[str, int]:
        policy = self._base_policy_row()
        action_dump = action.model_dump()
        for action_key, policy_key in ACTION_TO_POLICY.items():
            max_val = _NPI_MAX_MAPPINGS[policy_key]
            policy[policy_key] = int(np.clip(action_dump[action_key], 0, max_val))
        return policy

    def _phase(self) -> str:
        current_date = self.episode_df.iloc[min(self.day, len(self.episode_df) - 1)][
            "date"
        ]
        if current_date < pd.Timestamp("2020-12-14"):
            return "pre_vaccine"
        if self.vaccinated_pct < 60.0:
            return "rollout"
        return "endemic"

    def _build_observation(
        self, re_value: float, gdp_value: float, stringency_value: float
    ) -> PandemicPolicyObservation:
        if len(self.infected_history) >= 8:
            prev = self.infected_history[0]
            weekly_growth = (self.I - prev) / max(prev, 1e-8)
        else:
            weekly_growth = 0.0

        new_cases_norm = max(self.I - self.prev_I, 0.0)

        return PandemicPolicyObservation(
            susceptible=float(self.S),
            infected=float(self.I),
            recovered=float(self.R),
            re=float(re_value),
            new_cases_norm=float(new_cases_norm),
            weekly_growth_rate=float(weekly_growth),
            gdp_index=float(gdp_value),
            stringency_index=float(stringency_value),
            vaccinated_pct=float(np.clip(self.vaccinated_pct, 0.0, 100.0)),
            fully_vaccinated_pct=float(np.clip(self.fully_vaccinated_pct, 0.0, 100.0)),
            day=int(self.day),
            phase=self._phase(),
        )

    def _compute_reward(
        self,
        re_value: float,
        gdp_value: float,
        stringency_value: float,
        action: PandemicPolicyAction,
    ) -> PandemicPolicyReward:
        health = float(np.clip(1.0 - (self.I / 0.01), 0.0, 1.0))
        control = float(np.clip((1.5 - re_value) / 1.5, 0.0, 1.0))
        economic = float(np.clip(gdp_value / 100.0, 0.0, 1.0))
        stability = float(
            np.clip(1.0 - abs(stringency_value - self.prev_stringency) / 80.0, 0.0, 1.0)
        )

        support = 0.0
        if stringency_value >= 60.0:
            support = 0.5 * (action.income_support / 2.0) + 0.5 * (
                action.debt_relief / 2.0
            )

        penalty = 0.0
        if self.I > 0.003:
            penalty += 0.15
        if self.same_action_streak >= 10:
            penalty += min(0.20, 0.02 * (self.same_action_streak - 9))

        reward_value = (
            0.40 * health
            + 0.15 * control
            + 0.25 * economic
            + 0.15 * stability
            + 0.05 * support
        )
        reward_value = float(np.clip(reward_value - penalty, 0.0, 1.0))

        return PandemicPolicyReward(
            value=reward_value,
            health_component=health,
            economic_component=economic,
            stability_component=stability,
            support_component=float(np.clip(support, 0.0, 1.0)),
            penalty_component=float(np.clip(penalty, 0.0, 1.0)),
        )

    def _set_task(self, task: str) -> None:
        if task not in TASKS:
            raise ValueError(f"unknown task '{task}'")
        cfg = TASKS[task]
        mask = (self.df["date"] >= pd.Timestamp(cfg.start_date)) & (
            self.df["date"] <= pd.Timestamp(cfg.end_date)
        )
        episode_df = self.df.loc[mask].reset_index(drop=True)
        if episode_df.empty:
            raise ValueError(f"no data available for task '{task}' date range")

        max_steps = min(cfg.max_steps, len(episode_df))

        self.task_name = task
        self.task_cfg = cfg
        self.episode_df = episode_df
        self.max_steps = max_steps

    def reset(
        self, task: str = "flatten_curve", seed: int | None = None
    ) -> PandemicPolicyObservation:
        if seed is not None:
            np.random.seed(seed)

        self._set_task(task)

        self.day = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.trajectory = []
        self.last_action = None
        self.same_action_streak = 0

        row = self.episode_df.iloc[0]
        self.S = float(row["S"])
        self.I = float(row["I"])
        self.R = float(row["R"])
        self.prev_I = self.I
        self.prev_stringency = float(row.get("stringency_index", 0.0) or 0.0)
        self.vaccinated_pct = float(
            row.get("people_vaccinated_per_hundred", 0.0) or 0.0
        )
        self.fully_vaccinated_pct = float(
            row.get("people_fully_vaccinated_per_hundred", 0.0) or 0.0
        )

        self.infected_history.clear()
        self.infected_history.append(self.I)

        re_value = self._compute_re(self.prev_stringency / 100.0, self.S)
        gdp_value = float(np.clip(row.get("GDP_scaled", 90.0) or 90.0, 0.0, 100.0))
        obs = self._build_observation(re_value, gdp_value, self.prev_stringency)
        self.last_observation = obs
        return obs

    def step(self, action: PandemicPolicyAction) -> StepResponse:
        if self.done:
            # OpenEnv-style no-op step if episode is already done.
            reward = PandemicPolicyReward(
                value=0.0,
                health_component=0.0,
                economic_component=0.0,
                stability_component=0.0,
                support_component=0.0,
                penalty_component=0.0,
            )
            return StepResponse(
                observation=self.last_observation,
                reward=reward,
                done=True,
                info={"message": "episode already finished"},
            )

        policy = self._action_to_policy(action)
        stringency = self._policy_to_stringency(policy)
        s_norm = stringency / 100.0

        nu_base = float(NU_TV[min(self.day, len(NU_TV) - 1)]) if len(NU_TV) else 0.0
        nu = float(
            np.clip(nu_base * (1.0 + action.vaccination_policy / 10.0), 0.0, 0.02)
        )

        self.prev_I = self.I
        S_new, I_new, R_new = self._step_sir(s_norm=s_norm, nu=nu)
        re_value = self._compute_re(s_norm, S_new)
        gdp_value = self._compute_gdp(stringency)

        action_dump = action.model_dump()
        if (
            self.last_action is not None
            and action_dump == self.last_action.model_dump()
        ):
            self.same_action_streak += 1
        else:
            self.same_action_streak = 0
        self.last_action = action

        # Vaccination proxy progression from policy + exogenous nu.
        self.vaccinated_pct = float(
            np.clip(
                self.vaccinated_pct
                + (action.vaccination_policy * 0.10)
                + (nu * 2000.0),
                0.0,
                100.0,
            )
        )
        self.fully_vaccinated_pct = float(
            np.clip(
                self.fully_vaccinated_pct
                + (action.vaccination_policy * 0.07)
                + (nu * 1400.0),
                0.0,
                self.vaccinated_pct,
            )
        )

        self.S, self.I, self.R = S_new, I_new, R_new
        self.infected_history.append(self.I)

        reward = self._compute_reward(re_value, gdp_value, stringency, action)
        self.cumulative_reward += reward.value

        self.day += 1
        self.done = self.day >= self.max_steps

        obs = self._build_observation(re_value, gdp_value, stringency)
        self.last_observation = obs

        self.trajectory.append(
            {
                "day": obs.day,
                "infected": obs.infected,
                "re": obs.re,
                "gdp_index": obs.gdp_index,
                "stringency_index": obs.stringency_index,
                "vaccinated_pct": obs.vaccinated_pct,
                "reward": reward.value,
            }
        )

        self.prev_stringency = stringency

        info = {
            "task": self.task_name,
            "max_steps": self.max_steps,
            "nu": nu,
            "stringency": stringency,
            "cumulative_reward": float(self.cumulative_reward),
        }

        return StepResponse(observation=obs, reward=reward, done=self.done, info=info)

    def state(self) -> StateResponse:
        if self.last_observation is None:
            raise RuntimeError("environment has not been reset")
        return StateResponse(
            task=self.task_name,
            day=self.day,
            max_steps=self.max_steps,
            done=self.done,
            cumulative_reward=float(self.cumulative_reward),
            last_observation=self.last_observation,
        )

    def grade(self, task: str | None = None) -> GradeResponse:
        task_name = task or self.task_name
        if task_name not in TASKS:
            raise ValueError(f"unknown task '{task_name}'")

        if not self.trajectory:
            score = 0.0
            breakdown = {"empty_trajectory": 0.0}
        else:
            traj = pd.DataFrame(self.trajectory)
            infected = traj["infected"].values
            re_vals = traj["re"].values
            gdp_vals = traj["gdp_index"].values
            vax_vals = traj["vaccinated_pct"].values
            str_vals = traj["stringency_index"].values

            if task_name == "flatten_curve":
                safe_rate = float(np.mean(infected < 0.003))
                control_rate = float(np.mean(re_vals < 1.20))
                score = 0.7 * safe_rate + 0.3 * control_rate
                breakdown = {
                    "safe_rate": safe_rate,
                    "control_rate": control_rate,
                }
            elif task_name == "balanced_response":
                health_rate = float(np.mean(re_vals < 1.20))
                economy_rate = float(np.mean(gdp_vals >= 92.0))
                stability_rate = 1.0
                if len(str_vals) > 1:
                    stability_rate = float(np.mean(np.abs(np.diff(str_vals)) <= 8.0))
                harmonic = 0.0
                if health_rate + economy_rate > 0:
                    harmonic = (2.0 * health_rate * economy_rate) / (
                        health_rate + economy_rate
                    )
                score = 0.8 * harmonic + 0.2 * stability_rate
                breakdown = {
                    "health_rate": health_rate,
                    "economy_rate": economy_rate,
                    "stability_rate": stability_rate,
                    "harmonic": harmonic,
                }
            else:
                n = len(traj)
                one_third = max(1, n // 3)
                pre = traj.iloc[:one_third]
                mid = traj.iloc[one_third : 2 * one_third]
                post = traj.iloc[2 * one_third :]

                pre_control = float(np.mean(pre["re"] < 1.0)) if len(pre) else 0.0
                if len(mid) > 1:
                    vax_progress = float(
                        np.clip(
                            (
                                mid["vaccinated_pct"].iloc[-1]
                                - mid["vaccinated_pct"].iloc[0]
                            )
                            / 25.0,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    vax_progress = float(np.clip(vax_vals[-1] / 25.0, 0.0, 1.0))
                post_economy = (
                    float(np.mean(post["gdp_index"] >= 94.0)) if len(post) else 0.0
                )
                overflow_avoidance = float(1.0 - np.mean(infected > 0.003))

                score = (
                    0.35 * pre_control
                    + 0.25 * vax_progress
                    + 0.25 * post_economy
                    + 0.15 * overflow_avoidance
                )
                breakdown = {
                    "pre_control": pre_control,
                    "vax_progress": vax_progress,
                    "post_economy": post_economy,
                    "overflow_avoidance": overflow_avoidance,
                }

        score = float(np.clip(score, 0.0, 1.0))
        threshold = TASKS[task_name].success_threshold
        return GradeResponse(
            task=task_name,
            score=score,
            success=score >= threshold,
            threshold=threshold,
            breakdown=breakdown,
        )


app = FastAPI(title="Pandemic Policy OpenEnv", version="1.0.0")
runtime = PandemicPolicyOpenEnv()


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    try:
        obs = runtime.reset(task=req.task, seed=req.seed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(
        task=runtime.task_name, observation=obs, max_steps=runtime.max_steps
    )


@app.post("/step", response_model=StepResponse)
def step(action: PandemicPolicyAction) -> StepResponse:
    try:
        return runtime.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    try:
        return runtime.state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/grade", response_model=GradeResponse)
def grade(req: GradeRequest) -> GradeResponse:
    try:
        return runtime.grade(task=req.task)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
