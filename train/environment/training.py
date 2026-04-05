import numpy as np
import pandas as pd
import json
import os
import warnings
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import gymnasium as gym
from train.environment.epidemic_env import (
    EpidemicEnv,
    preprocess_for_env,
    BETA,
    GAMMA,
    NU_TV,
    GDP_COEFFS,
    _LOOKBACK,
    _NPI_MAX_MAPPINGS,
)

warnings.filterwarnings("ignore")

CFG = dict(
    # --- data ---sir_data = pd.read_csv(r"data\combined_us_data.csv")
    __COUNTRY__="United States",
    _RECOVERY_DAYS=14,
    data_path=r"data\combined_us_data.csv",
    country="United States",
    date_start="2022-12-31",
    date_end="2020-12-20",
    N=330_000_000,
    recovery_days=14,
    # --- env ---
    lookback=_LOOKBACK,
    # --- PPO hyperparameters ---
    total_timesteps=500_000,  # increase to 1M+ for production
    n_envs=4,  # parallel envs for faster sampling
    n_steps=512,  # rollout buffer size per env
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # entropy bonus → encourages exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    learning_rate=3e-4,
    # --- network ---
    lstm_hidden=64,
    lstm_layers=1,
    fc_hidden=64,
    shared_dim=128,  # dimension after concatenation
    # --- saving ---
    log_dir="./train/environment/logs/",
    model_dir="./train/environment/models/",
    fig_dir="./train/environment/figures/",
)

os.makedirs(CFG["log_dir"], exist_ok=True)
os.makedirs(CFG["model_dir"], exist_ok=True)
os.makedirs(CFG["fig_dir"], exist_ok=True)


class EpidemicFeaturesExtractor(BaseFeaturesExtractor):
    """
    Dict observation → shared latent vector.

    observation_space keys:
        'time_series' : (lookback, n_ts_features)
        'static'      : (n_static_features,)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        fc_hidden: int = 64,
        shared_dim: int = 128,
    ):

        # features_dim = size of the final concatenated vector
        super().__init__(observation_space, features_dim=shared_dim)

        ts_shape = observation_space["time_series"].shape  # (lookback, n_ts)
        static_shape = observation_space["static"].shape  # (n_static,)

        n_ts = ts_shape[1]
        n_static = static_shape[0]

        # --- LSTM branch ---
        self.lstm = nn.LSTM(
            input_size=n_ts,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # --- FCN branch ---
        self.fcn = nn.Sequential(
            nn.Linear(n_static, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(),
        )

        # --- merge layer ---
        merged_dim = lstm_hidden + fc_hidden
        self.merge = nn.Sequential(
            nn.Linear(merged_dim, shared_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        ts = obs["time_series"]  # (batch, lookback, n_ts)
        static = obs["static"]  # (batch, n_static)

        # LSTM: take only the last hidden state
        _, (h_n, _) = self.lstm(ts)
        lstm_out = h_n[-1]  # (batch, lstm_hidden)

        # FCN
        fc_out = self.fcn(static)  # (batch, fc_hidden)

        # Concatenate and project
        merged = torch.cat([lstm_out, fc_out], dim=1)
        return self.merge(merged)  # (batch, shared_dim)


# ============================================================
# 3. REWARD TRACKING CALLBACK
# ============================================================
class RewardTrackingCallback(BaseCallback):
    """
    Logs per-episode reward, Re, GDP, and infected fraction
    so we can plot training curves afterwards.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_Re = []
        self.episode_gdp = []
        self.episode_infected = []
        self._ep_reward = 0.0
        self._ep_Re = []
        self._ep_gdp = []
        self._ep_infected = []

    def _on_step(self) -> bool:
        # accumulate reward across the vectorised envs
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]

        self._ep_reward += float(np.mean(rewards))

        for info in infos:
            if "Re" in info:
                self._ep_Re.append(info["Re"])
            if "gdp" in info:
                self._ep_gdp.append(info["gdp"])
            if "I" in info:
                self._ep_infected.append(info["I"])

            # episode finished signal
            if "episode" in info:
                self.episode_rewards.append(self._ep_reward)
                self.episode_Re.append(np.mean(self._ep_Re) if self._ep_Re else np.nan)
                self.episode_gdp.append(
                    np.mean(self._ep_gdp) if self._ep_gdp else np.nan
                )
                self.episode_infected.append(
                    np.mean(self._ep_infected) if self._ep_infected else np.nan
                )

                # reset accumulators
                self._ep_reward = 0.0
                self._ep_Re = []
                self._ep_gdp = []
                self._ep_infected = []

        return True  # continue training

    def save_logs(self, path: str):
        logs = pd.DataFrame(
            {
                "episode": range(len(self.episode_rewards)),
                "total_reward": self.episode_rewards,
                "mean_Re": self.episode_Re,
                "mean_gdp": self.episode_gdp,
                "mean_infected": self.episode_infected,
            }
        )
        logs.to_csv(path, index=False)
        print(f"  Training logs saved → {path}")


# ============================================================
# 5. ENVIRONMENT FACTORY
# ============================================================
def make_env(df: pd.DataFrame, training: bool = True):
    """Returns a callable that creates one monitored environment."""

    def _init():
        env = EpidemicEnv(
            df=df,
            beta=BETA,
            gamma=GAMMA,
            nu_tv=NU_TV,
            gdp_coeffs=GDP_COEFFS,
            lookback=CFG["lookback"],
            training=training,
        )
        env = Monitor(env, CFG["log_dir"])
        return env

    return _init


# ============================================================
# 6. BUILD PPO MODEL
# ============================================================
def build_model(env) -> PPO:
    policy_kwargs = dict(
        features_extractor_class=EpidemicFeaturesExtractor,
        features_extractor_kwargs=dict(
            lstm_hidden=CFG["lstm_hidden"],
            lstm_layers=CFG["lstm_layers"],
            fc_hidden=CFG["fc_hidden"],
            shared_dim=CFG["shared_dim"],
        ),
        # policy and value networks on top of shared extractor
        net_arch=dict(
            pi=[128, 64],
            vf=[128, 64],
        ),
        activation_fn=nn.ReLU,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=CFG["learning_rate"],
        n_steps=CFG["n_steps"],
        batch_size=CFG["batch_size"],
        n_epochs=CFG["n_epochs"],
        gamma=CFG["gamma"],
        gae_lambda=CFG["gae_lambda"],
        clip_range=CFG["clip_range"],
        ent_coef=CFG["ent_coef"],
        vf_coef=CFG["vf_coef"],
        max_grad_norm=CFG["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=CFG["log_dir"],
        verbose=1,
    )

    print("\nModel architecture:")
    print(model.policy)
    return model


# ============================================================
# 7. TRAINING
# ============================================================
def train(model: PPO, eval_env, reward_cb: RewardTrackingCallback) -> PPO:

    # Save a checkpoint every 50k steps
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=CFG["model_dir"],
        name_prefix="ppo_epidemic_us",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=CFG["model_dir"],
        log_path=CFG["log_dir"],
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        warn=False,
    )

    print("\n" + "=" * 55)
    print("TRAINING")
    print(f"  total_timesteps : {CFG['total_timesteps']:,}")
    print(f"  n_envs          : {CFG['n_envs']}")
    print(f"  n_steps         : {CFG['n_steps']}")
    print(f"  batch_size      : {CFG['batch_size']}")
    print("=" * 55)

    model.learn(
        total_timesteps=CFG["total_timesteps"],
        callback=[reward_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    model.save(os.path.join(CFG["model_dir"], "ppo_epidemic_us_final"))
    print("\n  Final model saved.")
    return model


# ============================================================
# 8. EVALUATION ROLLOUT
# ============================================================
def evaluate_rollout(model: PPO, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run one deterministic episode and record everything
    for comparison plots.
    """
    print("\n" + "=" * 55)
    print("EVALUATION ROLLOUT")
    print("=" * 55)

    env = EpidemicEnv(
        df=df,
        beta=BETA,
        gamma=GAMMA,
        nu_tv=NU_TV,
        gdp_coeffs=GDP_COEFFS,
        lookback=CFG["lookback"],
        training=False,
    )

    obs, _ = env.reset()
    done = False
    records = []
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # decode action back to NPI dict
        action_dict = {k: int(action[i]) for i, k in enumerate(_NPI_MAX_MAPPINGS)}
        from epidemic_env import npis_to_stringency

        stringency = npis_to_stringency(action_dict)

        records.append(
            {
                "step": env.t,
                "date": df.iloc[min(env.t, len(df) - 1)]["date"],
                "S_rl": info["S"],
                "I_rl": info["I"],
                "R_rl": info["R"],
                "Re_rl": info["Re"],
                "gdp_rl": info["gdp"],
                "stringency_rl": stringency,
                "reward": reward,
                **{f"action_{k}": v for k, v in action_dict.items()},
            }
        )

    print(f"  Steps completed : {len(records)}")
    print(f"  Total reward    : {total_reward:,.2f}")

    rollout_df = pd.DataFrame(records)
    rollout_df.to_csv(os.path.join(CFG["log_dir"], "rollout_us.csv"), index=False)
    print("  Rollout saved → logs/rollout_us.csv")
    return rollout_df


# ============================================================
# 9. PLOTTING
# ============================================================
def plot_training_curves(reward_cb: RewardTrackingCallback):
    logs = pd.DataFrame(
        {
            "episode": range(len(reward_cb.episode_rewards)),
            "total_reward": reward_cb.episode_rewards,
            "mean_Re": reward_cb.episode_Re,
            "mean_gdp": reward_cb.episode_gdp,
            "mean_infected": reward_cb.episode_infected,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("PPO Training Curves — US Epidemic Control")

    axes[0, 0].plot(logs["episode"], logs["total_reward"], lw=1)
    axes[0, 0].set_title("Total Reward per Episode")
    axes[0, 0].set_xlabel("Episode")

    axes[0, 1].plot(logs["episode"], logs["mean_Re"], color="red", lw=1)
    axes[0, 1].axhline(1.0, ls="--", color="black", lw=0.8)
    axes[0, 1].axhline(1.25, ls="--", color="orange", lw=0.8)
    axes[0, 1].axhline(1.5, ls="--", color="red", lw=0.8)
    axes[0, 1].set_title("Mean Re per Episode")
    axes[0, 1].set_xlabel("Episode")

    axes[1, 0].plot(logs["episode"], logs["mean_gdp"], color="green", lw=1)
    axes[1, 0].set_title("Mean GDP per Episode")
    axes[1, 0].set_xlabel("Episode")

    axes[1, 1].plot(logs["episode"], logs["mean_infected"], color="orange", lw=1)
    axes[1, 1].axhline(0.003, ls="--", color="red", lw=0.8, label="threshold=0.003")
    axes[1, 1].set_title("Mean Infected Fraction per Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].legend()

    plt.tight_layout()
    path = os.path.join(CFG["fig_dir"], "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")


def plot_rollout_vs_actual(rollout_df: pd.DataFrame, df: pd.DataFrame):
    """
    Compare RL policy trajectory against actual historical data.
    Mirrors the paper's Figure 11 / Figure 12 layout.
    """
    from scipy.signal import medfilt

    # Smooth RL stringency (median filter as in paper)
    rollout_df["stringency_rl_smooth"] = medfilt(
        rollout_df["stringency_rl"], kernel_size=7
    )

    # Align dates
    df_trim = df.iloc[CFG["lookback"] : CFG["lookback"] + len(rollout_df)].copy()
    df_trim = df_trim.reset_index(drop=True)

    dates_actual = df_trim["date"].values
    dates_rl = rollout_df["date"].values

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("RL Policy vs Actual — United States", fontsize=14)

    # (a) Stringency
    axes[0, 0].plot(dates_actual, df_trim["stringency_index"], label="Actual", lw=1.5)
    axes[0, 0].plot(
        dates_rl, rollout_df["stringency_rl_smooth"], label="RL", lw=1.5, ls="--"
    )
    axes[0, 0].set_title("(a) Stringency over Time")
    axes[0, 0].set_ylabel("Stringency Index")
    axes[0, 0].legend()

    # (b) SIR dynamics
    axes[0, 1].plot(dates_actual, df_trim["S"] * 100, "b-", label="S actual", lw=1.2)
    axes[0, 1].plot(dates_actual, df_trim["I"] * 100, "r-", label="I actual", lw=1.2)
    axes[0, 1].plot(dates_actual, df_trim["R"] * 100, "g-", label="R actual", lw=1.2)
    axes[0, 1].plot(dates_rl, rollout_df["S_rl"] * 100, "b--", label="S rl", lw=1.2)
    axes[0, 1].plot(dates_rl, rollout_df["I_rl"] * 100, "r--", label="I rl", lw=1.2)
    axes[0, 1].plot(dates_rl, rollout_df["R_rl"] * 100, "g--", label="R rl", lw=1.2)
    axes[0, 1].set_title("(b) SIR Dynamics")
    axes[0, 1].set_ylabel("% of Population")
    axes[0, 1].legend(fontsize=7)

    # (c) Infected zoomed
    axes[1, 0].plot(dates_actual, df_trim["I"], "r-", label="I actual", lw=1.5)
    axes[1, 0].plot(dates_rl, rollout_df["I_rl"], "r--", label="I rl", lw=1.5)
    axes[1, 0].axhline(0.003, ls="--", color="black", label="threshold", lw=0.8)
    axes[1, 0].set_title("(c) Infected Population")
    axes[1, 0].set_ylabel("Proportion")
    axes[1, 0].legend()

    # (d) GDP
    axes[1, 1].plot(
        dates_actual, df_trim["GDP_scaled"], "g-", label="GDP actual", lw=1.5
    )
    axes[1, 1].plot(dates_rl, rollout_df["gdp_rl"], "g--", label="GDP rl", lw=1.5)
    axes[1, 1].set_title("(d) Normalized GDP")
    axes[1, 1].set_ylabel("GDP scaled")
    axes[1, 1].legend()

    # (e) Re
    axes[2, 0].plot(
        dates_actual, df_trim["Re"], color="purple", label="Re actual", lw=1.5
    )
    axes[2, 0].plot(
        dates_rl, rollout_df["Re_rl"], color="purple", ls="--", label="Re rl", lw=1.5
    )
    axes[2, 0].axhline(1.0, ls=":", color="black", lw=0.8)
    axes[2, 0].axhline(1.25, ls=":", color="orange", lw=0.8)
    axes[2, 0].axhline(1.5, ls=":", color="red", lw=0.8)
    axes[2, 0].set_title("(e) Effective Reproduction Number")
    axes[2, 0].set_ylabel("Re")
    axes[2, 0].legend()

    # (f) Cumulative reward
    axes[2, 1].plot(
        dates_rl,
        rollout_df["reward"].cumsum(),
        color="blue",
        lw=1.5,
        label="RL cumulative reward",
    )
    axes[2, 1].set_title(f"(f) Cumulative Reward = {rollout_df['reward'].sum():,.1f}")
    axes[2, 1].set_ylabel("Cumulative Reward")
    axes[2, 1].legend()

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    path = os.path.join(CFG["fig_dir"], "rollout_vs_actual.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")


def main():

    print("Preparing data...")
    sir_data = pd.read_csv(CFG["data_path"], low_memory=False)
    sir_data["date"] = pd.to_datetime(sir_data["date"])
    sir_data = sir_data.sort_values("date").reset_index(drop=True)
    sir_data["total_recovered"] = (
        sir_data["total_cases"].shift(CFG["_RECOVERY_DAYS"]).fillna(0)
        - sir_data["total_deaths"]
    ).clip(lower=0)
    sir_data["active_infected"] = (
        sir_data["total_cases"] - sir_data["total_recovered"] - sir_data["total_deaths"]
    ).clip(lower=0)

    sir_data["S_count"] = CFG["N"] - sir_data["total_cases"]
    sir_data["I_count"] = sir_data["active_infected"]
    sir_data["R_count"] = sir_data["total_recovered"]
    sir_data["S"] = sir_data["S_count"] / CFG["N"]
    sir_data["I"] = sir_data["I_count"] / CFG["N"]
    sir_data["R"] = sir_data["R_count"] / CFG["N"]
    sir_data["S_pct"] = sir_data["S"] * 100
    sir_data["I_pct"] = sir_data["I"] * 100
    sir_data["R_pct"] = sir_data["R"] * 100
    sir_data["s_norm"] = (
        sir_data["stringency_index"].fillna(method="ffill").fillna(0) / 100.0
    )
    sir_data["nu"] = sir_data["daily_vaccinations_per_million"].fillna(0) / 1_000_000
    print(f"  Shape: {sir_data.shape}")

    # --- vectorised training env ---
    train_env = make_vec_env(
        make_env(sir_data, training=True),
        n_envs=CFG["n_envs"],
        seed=42,
    )
    # normalise observations and rewards for more stable training
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=500.0,
    )

    eval_env = make_vec_env(
        make_env(sir_data, training=False),
        n_envs=1,
        seed=42,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # never normalise reward on eval
        clip_obs=10.0,
        training=False,  # freeze running stats during eval
    )

    # --- model ---
    model = build_model(train_env)
    reward_cb = RewardTrackingCallback()

    # --- train ---
    model = train(model, eval_env, reward_cb)

    # --- save VecNormalize stats (needed for deployment) ---
    train_env.save(os.path.join(CFG["model_dir"], "vec_normalize_us.pkl"))

    # --- logs ---
    reward_cb.save_logs(os.path.join(CFG["log_dir"], "training_logs.csv"))

    # --- evaluate ---
    rollout_df = evaluate_rollout(model, sir_data)

    # --- plots ---
    plot_training_curves(reward_cb)
    plot_rollout_vs_actual(rollout_df, sir_data)

    # --- final summary ---
    print("\n" + "=" * 55)
    print("DONE")
    print("=" * 55)
    print(f"  Best model   → {CFG['model_dir']}best_model.zip")
    print(f"  Final model  → {CFG['model_dir']}ppo_epidemic_us_final.zip")
    print(f"  VecNormalize → {CFG['model_dir']}vec_normalize_us.pkl")
    print(f"  Rollout CSV  → {CFG['log_dir']}rollout_us.csv")
    print(f"  Figures      → {CFG['fig_dir']}")


if __name__ == "__main__":
    main()
