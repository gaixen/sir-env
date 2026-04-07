import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import logging
import os
from train.fitting.commons import (
    loss_sir,
    loss_i,
    ode_lockdown_nu,
    integrate,
)

sir_data = pd.read_csv(r"data\combined_us_data.csv")
__COUNTRY__ = "United States"
_POPULATION = 330_000_000
_DATE_START = sir_data.date.min()
_DATE_END = sir_data.date.max()
_RECOVERY_DAYS = 14
WINDOW_LENGTHS = list(range(5, 55, 5))  # [5,10,...,50]
OPTIMAL_WINDOW = 15

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
_PATH = os.path.dirname(os.path.abspath(__file__))

sir_data["date"] = pd.to_datetime(sir_data["date"])
sir_data = sir_data.sort_values("date").reset_index(drop=True)
sir_data["total_recovered"] = (
    sir_data["total_cases"].shift(_RECOVERY_DAYS).fillna(0) - sir_data["total_deaths"]
).clip(lower=0)
sir_data["active_infected"] = (
    sir_data["total_cases"] - sir_data["total_recovered"] - sir_data["total_deaths"]
).clip(lower=0)

sir_data["S_count"] = _POPULATION - sir_data["total_cases"]
sir_data["I_count"] = sir_data["active_infected"]
sir_data["R_count"] = sir_data["total_recovered"]
sir_data["S"] = sir_data["S_count"] / _POPULATION
sir_data["I"] = sir_data["I_count"] / _POPULATION
sir_data["R"] = sir_data["R_count"] / _POPULATION
sir_data["S_pct"] = sir_data["S"] * 100
sir_data["I_pct"] = sir_data["I"] * 100
sir_data["R_pct"] = sir_data["R"] * 100
sir_data["s_norm"] = (
    sir_data["stringency_index"].fillna(method="ffill").fillna(0) / 100.0
)
sir_data["nu"] = sir_data["daily_vaccinations_per_million"].fillna(0) / 1_000_000


def compute_nu_for_window(
    window_len, beta_fixed, gamma_fixed, S_data, I_data, R_data, s_arr
):
    """Estimate ν for each sub-window; return full-length array."""
    T = len(S_data)
    nu_seg = []
    start = 0

    while start < T:
        end = min(start + window_len, T)
        seg_len = end - start

        S_loc = S_data[start:end]
        I_loc = I_data[start:end]
        R_loc = R_data[start:end]
        s_loc = s_arr[start:end]
        y0_loc = [S_loc[0], I_loc[0], R_loc[0]]

        def obj(params):
            nu = params[0]
            if nu < 0:
                return 1e12
            sol = integrate(
                ode_lockdown_nu,
                y0_loc,
                seg_len,
                args=(beta_fixed, gamma_fixed, nu, s_loc),
            )
            # Combined loss: SIR + I only (balanced as per paper)
            li = loss_sir(S_loc, I_loc, R_loc, sol[:, 0], sol[:, 1], sol[:, 2]) + loss_i(
                I_loc, sol[:, 1]
            )
            return li

        res = minimize(
            obj,
            x0=[1e-4],
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10},
        )
        nu_val = max(float(res.x[0]), 0.0)  # enforce ν ≥ 0
        nu_seg.append((seg_len, nu_val))
        start += window_len

    # Expand segments back to daily array
    nu_full = np.concatenate([np.full(seg_len, nu_val) for seg_len, nu_val in nu_seg])
    return nu_full[:T]


def window_search(beta_c, gamma_c, S_data, I_data, R_data, s_arr):
    print("\n" + "=" * 55)
    print("MODEL 2d: Window search for time-varying ν")
    print("=" * 55)

    T = len(S_data)
    y0 = [S_data[0], I_data[0], R_data[0]]

    records = []
    nu_arrays = {}

    for wl in WINDOW_LENGTHS:
        nu_full = compute_nu_for_window(
            wl, beta_c, gamma_c, S_data, I_data, R_data, s_arr
        )

        sol = integrate(ode_lockdown_nu, y0, T, args=(beta_c, gamma_c, nu_full, s_arr))

        l_sir = loss_sir(S_data, I_data, R_data, sol[:, 0], sol[:, 1], sol[:, 2])
        l_i = loss_i(I_data, sol[:, 1])

        records.append({"window": wl, "loss_sir": l_sir, "loss_i": l_i})
        nu_arrays[wl] = nu_full

        print(f"  window={wl:3d} | loss_SIR={l_sir:>15,.1f} | loss_I={l_i:>13,.1f}")

    # Paper uses window=15 as the sweet spot
    best_nu = nu_arrays[OPTIMAL_WINDOW]
    print(f"\n  → Using window = {OPTIMAL_WINDOW} days")
    print(
        f"  ν stats: mean={best_nu.mean():.5f}, "
        f"max={best_nu.max():.5f}, "
        f"mode≈{float(pd.Series(best_nu.round(5)).mode()[0]):.5f}"
    )

    return pd.DataFrame(records), nu_arrays, best_nu


def plot_window_search(window_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Window Length Search — United States", fontsize=13)

    axes[0].plot(window_df["window"], window_df["loss_sir"], "bo-", lw=1.5)
    axes[0].axvline(
        OPTIMAL_WINDOW, color="red", ls="--", label=f"chosen={OPTIMAL_WINDOW}"
    )
    axes[0].set_title("loss_SIR vs Window Length")
    axes[0].set_xlabel("Window Length (days)")
    axes[0].legend()

    axes[1].plot(window_df["window"], window_df["loss_i"], "go-", lw=1.5)
    axes[1].axvline(
        OPTIMAL_WINDOW, color="red", ls="--", label=f"chosen={OPTIMAL_WINDOW}"
    )
    axes[1].set_title("loss_I vs Window Length")
    axes[1].set_xlabel("Window Length (days)")
    axes[1].legend()

    plt.tight_layout()
    # Image saving disabled to prevent writing files during automated runs
    logger.info("Skipping saving figure: window_search_US.png")
    plt.show()



