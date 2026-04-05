import numpy as np
import pandas as pd
import json
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import odeint

with open("./train/sir_params_US.json", "r") as f:
    params = json.load(f)

BETA = params["beta_final"]
GAMMA = params["gamma_final"]
NU_TV = np.array(params["nu_tv"])
GDP_COEFFS = np.array(params["gdp_coeffs"])
N = 330_000_000


def compute_gdp(stringency: float, coeffs: np.ndarray) -> float:
    """Evaluate cubic polynomial at given stringency."""
    p = np.poly1d(coeffs)
    return float(np.clip(p(stringency), 0, 100))


def compute_Re(beta: float, gamma: float, s_norm: float, S: float) -> float:
    """Effective reproduction number."""
    return beta * (1.0 - s_norm) * S / gamma


def ode_lockdown_nu(y, t, beta, gamma, nu, s):
    S, Inf, R = y
    eff = beta * (1.0 - s)
    dS = -eff * S * Inf - nu * S
    dI = eff * S * Inf - gamma * Inf
    dR = gamma * Inf + nu * S
    return [dS, dI, dR]


def step_sir(
    S: float,
    Inf: float,
    R: float,
    beta: float,
    gamma: float,
    nu: float,
    s_norm: float,
    n_steps: int = 1,
) -> tuple:
    """Integrate SIR one day forward."""
    y0 = [S, Inf, R]
    t = np.linspace(0, n_steps, n_steps + 1)
    sol = odeint(ode_lockdown_nu, y0, t, args=(beta, gamma, nu, s_norm), hmax=1.0)
    return float(sol[-1, 0]), float(sol[-1, 1]), float(sol[-1, 2])


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

NPI_KEYS = list(_NPI_MAX_MAPPINGS.keys())
NPI_DIMS = [v + 1 for v in _NPI_MAX_MAPPINGS.values()]  # number of levels per action


def npis_to_stringency(action_dict: dict) -> float:
    """
    Convert individual NPI levels to a single stringency score [0, 100].
    Economic support columns (e1, e2) excluded from stringency
    but kept as actions for reward shaping later.
    """
    stringency_keys = [k for k in NPI_KEYS if not k.startswith("e")]
    components = []
    for k in stringency_keys:
        val = action_dict[k]
        max_val = _NPI_MAX_MAPPINGS[k]
        components.append((val / max_val) * 100.0)
    return float(np.mean(components))


_LOOKBACK = 14  # 2 weeks of history


class EpidemicEnv(gym.Env):
    """
    US COVID-19 Epidemic Control Environment.

    Observation (Dict):
        time_series : (LOOKBACK, n_ts_features)  → fed to LSTM
        static      : (n_static_features,)        → fed to FCN

    Action (MultiDiscrete):
        One level per NPI column in NPI_KEYS order.
    """

    metadata = {"render_modes": []}

    # ------ feature lists ------
    TS_FEATURES = [
        "stringency",
        "gdp_scaled",
        "re",
        "new_cases_norm",
        "new_deaths_norm",
        "weekly_pct_growth_cases",
        "cfr_short_term",
    ]

    STATIC_FEATURES = [
        "S",
        "I",
        "R",
        "people_vaccinated_per_hundred",
        "share_doses_used",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        beta: float = BETA,
        gamma: float = GAMMA,
        nu_tv: np.ndarray = NU_TV,
        gdp_coeffs: np.ndarray = GDP_COEFFS,
        lookback: int = _LOOKBACK,
        start_date: str = "2020-05-01",
        end_date: str = "2022-10-31",
        training: bool = True,
    ):

        super().__init__()

        self.df = df.copy().reset_index(drop=True)
        self.beta = beta
        self.gamma = gamma
        self.nu_tv = nu_tv
        self.gdp_coeffs = gdp_coeffs
        self.lookback = lookback
        self.training = training
        self.t_min = lookback
        self.t_max = len(df) - 1
        self.action_space = spaces.MultiDiscrete(NPI_DIMS)
        n_ts = len(self.TS_FEATURES)
        n_static = len(self.STATIC_FEATURES)

        self.observation_space = spaces.Dict(
            {
                "time_series": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(lookback, n_ts), dtype=np.float32
                ),
                "static": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_static,), dtype=np.float32
                ),
            }
        )

        self.t = None
        self.S = None
        self.I = None
        self.R = None
        self.history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = self.t_min

        row = self.df.iloc[self.t]
        self.S = float(row["S"])
        self.I = float(row["I"])
        self.R = float(row["R"])
        self.history = self._build_initial_history()

        obs = self._get_obs()
        info = {}
        return obs, info

    def _build_initial_history(self) -> np.ndarray:
        """Fill the lookback window from real historical data."""
        buf = np.zeros((self.lookback, len(self.TS_FEATURES)), dtype=np.float32)

        for i, t in enumerate(range(self.t - self.lookback, self.t)):
            buf[i] = self._ts_row(t)
        return buf

    def _ts_row(self, t: int) -> np.ndarray:
        """Extract time-series features at timestep t from dataframe."""
        row = self.df.iloc[t]
        s = float(row.get("stringency_index", 0) or 0)
        return np.array(
            [
                s / 100.0,
                float(row.get("GDP_scaled", 100) or 100) / 100.0,
                float(row.get("Re", 1.0) or 1.0),
                float(row.get("new_cases_norm", 0) or 0),
                float(row.get("new_deaths_norm", 0) or 0),
                float(row.get("weekly_pct_growth_cases", 0) or 0) / 100.0,
                float(row.get("cfr_short_term", 0) or 0),
            ],
            dtype=np.float32,
        )

    def _get_obs(self) -> dict:
        static = np.array(
            [
                self.S,
                self.I,
                self.R,
                float(self.df.iloc[self.t].get("people_vaccinated_per_hundred", 0) or 0)
                / 100.0,
                float(self.df.iloc[self.t].get("share_doses_used", 0) or 0),
            ],
            dtype=np.float32,
        )

        return {
            "time_series": self.history.copy(),
            "static": static,
        }

    def step(self, action: np.ndarray):

        action_dict = {k: int(action[i]) for i, k in enumerate(NPI_KEYS)}
        stringency = npis_to_stringency(action_dict)
        s_norm = stringency / 100.0

        # --- get exogenous vaccination rate ---
        nu = float(self.nu_tv[min(self.t, len(self.nu_tv) - 1)])

        # --- integrate SIR one step ---
        S_new, I_new, R_new = step_sir(
            self.S, self.I, self.R, self.beta, self.gamma, nu, s_norm
        )

        # --- compute derived quantities ---
        Re = compute_Re(self.beta, self.gamma, s_norm, S_new)
        gdp = compute_gdp(stringency, self.gdp_coeffs)
        gdp_norm = gdp / 100.0

        # --- reward ---
        reward = self._compute_reward(
            Re=Re,
            gdp_norm=gdp_norm,
            I_new=I_new,
            stringency_prev=npis_to_stringency(
                {k: int(self.df.iloc[self.t].get(k, 0) or 0) for k in NPI_KEYS}
            ),
            stringency_curr=stringency,
            action_dict=action_dict,
        )

        # --- update internal state ---
        self.S = S_new
        self.I = I_new
        self.R = R_new
        self.t += 1

        # --- update history buffer (roll and append) ---
        new_ts_row = np.array(
            [
                s_norm,
                gdp_norm,
                Re,
                I_new,  # proxy for new_cases_norm
                0.0,  # new_deaths (not tracked in SIR)
                0.0,  # weekly_pct_growth (compute if needed)
                0.0,  # cfr_short_term (compute if needed)
            ],
            dtype=np.float32,
        )

        self.history = np.roll(self.history, shift=-1, axis=0)
        self.history[-1] = new_ts_row

        # --- termination ---
        terminated = self.t >= self.t_max
        truncated = False

        obs = self._get_obs()
        info = {
            "Re": Re,
            "gdp": gdp,
            "S": S_new,
            "I": I_new,
            "R": R_new,
            "stringency": stringency,
        }

        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------------
    # REWARD
    # ----------------------------------------------------------
    def _compute_reward(
        self,
        Re: float,
        gdp_norm: float,
        I_new: float,
        stringency_prev: float,
        stringency_curr: float,
        action_dict: dict,
    ) -> float:

        # --- core reward (mirrors paper) ---
        if Re > 1.5:
            r_core = -20.0 * Re
        elif 1.25 <= Re <= 1.5:
            r_core = 100.0 * gdp_norm
        else:  # Re < 1.25
            r_core = 200.0 * gdp_norm

        # --- infection threshold penalty ---
        INFECTED_THRESHOLD = 0.003
        r_infection = 50.0 if I_new < INFECTED_THRESHOLD else -2000.0

        # --- stability penalty: penalise large stringency swings ---
        r_stability = -12.0 * abs(stringency_curr - stringency_prev)

        # --- economic support bonus ---
        # Reward income support and debt relief during high stringency
        econ_bonus = 0.0
        if stringency_curr > 60:
            econ_bonus += action_dict.get("e1_income_support", 0) * 10.0
            econ_bonus += action_dict.get("e2_debt_contract_relief", 0) * 5.0

        reward = r_core + r_infection + r_stability + econ_bonus

        return float(reward)

    # ----------------------------------------------------------
    # RENDER (optional)
    # ----------------------------------------------------------
    def render(self):
        print(f"t={self.t:4d} | S={self.S:.4f} I={self.I:.6f} R={self.R:.4f}")


# ============================================================
# 5. PREPROCESSING: ADD DERIVED COLUMNS TO DATAFRAME
# ============================================================
def preprocess_for_env(df: pd.DataFrame, beta: float, gamma: float) -> pd.DataFrame:
    """
    Add Re, new_cases_norm, new_deaths_norm columns
    so the environment can read them from the dataframe.
    """
    df = df.copy()

    # Normalise case/death counts to [0,1] range
    df["new_cases_norm"] = df["new_cases"].fillna(0) / df["new_cases"].max()
    df["new_deaths_norm"] = df["new_deaths"].fillna(0) / df["new_deaths"].max()

    # Re from observed stringency and S proportion
    df["Re"] = beta * (1 - df["stringency_index"].fillna(0) / 100.0) * df["S"] / gamma

    # Forward-fill any remaining NaN in NPI columns
    npi_cols = list(_NPI_MAX_MAPPINGS.keys())
    df[npi_cols] = df[npi_cols].fillna(method="ffill").fillna(0)

    return df


def sanity_check(env: EpidemicEnv):
    print("\n" + "=" * 55)
    print("SANITY CHECK: Random policy rollout")
    print("=" * 55)

    obs, _ = env.reset()
    print(f"  time_series shape : {obs['time_series'].shape}")
    print(f"  static shape      : {obs['static'].shape}")

    total_reward = 0.0
    for step_idx in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"  step={step_idx + 1:2d} | "
            f"Re={info['Re']:.3f} | "
            f"GDP={info['gdp']:.2f} | "
            f"I={info['I']:.6f} | "
            f"stringency={info['stringency']:.1f} | "
            f"reward={reward:.2f}"
        )
        if terminated:
            break

    print(f"\n  Total reward over 10 steps: {total_reward:.2f}")
    print("  Environment working correctly ✓")


if __name__ == "__main__":
    # Load your preprocessed US dataframe
    df_raw = pd.read_csv(r"data\combined_us_data.csv", low_memory=False)
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    us = df_raw[df_raw["country"] == "United States"].copy()
    us = us[(us["date"] >= "2020-05-01") & (us["date"] <= "2022-10-31")]
    us = us.sort_values("date").reset_index(drop=True)

    # Add SIR compartments (reuse function from SIR script)
    N = 330_000_000
    RECOVERY_DAYS = 14
    us["total_cases"] = us["total_cases"].fillna(method="ffill").fillna(0)
    us["total_deaths"] = us["total_deaths"].fillna(method="ffill").fillna(0)
    us["total_recovered"] = (
        us["total_cases"].shift(RECOVERY_DAYS).fillna(0) - us["total_deaths"]
    ).clip(lower=0)
    us["active_infected"] = (
        us["total_cases"] - us["total_recovered"] - us["total_deaths"]
    ).clip(lower=0)
    us["S"] = (N - us["total_cases"]) / N
    us["I"] = us["active_infected"] / N
    us["R"] = us["total_recovered"] / N

    # Add derived columns
    us = preprocess_for_env(us, beta=BETA, gamma=GAMMA)

    # Build environment
    env = EpidemicEnv(
        df=us,
        beta=BETA,
        gamma=GAMMA,
        nu_tv=NU_TV,
        gdp_coeffs=GDP_COEFFS,
        lookback=_LOOKBACK,
    )

    # Sanity check
    sanity_check(env)

    print("\n  Environment ready.")
    print("  Next step → train RL agent (Step 5)")
