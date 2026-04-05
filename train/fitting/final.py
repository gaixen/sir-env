import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import logging
import json
import os
from train.fitting.commons import (
    loss_sir,
    loss_i,
    ode_lockdown_nu,
    integrate,
)
from train.fitting.simple_sir import fit_simple_sir
from train.fitting.gdp_fitting import fit_gdp_polynomial, plot_gdp_stringency
from train.fitting.sir_lockdown import fit_sir_lockdown, fit_sir_lockdown_const_nu, plot_nu_time_series
from train.fitting.window_search import window_search, plot_window_search

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

def fit_final_model(S_data, I_data, R_data, s_arr, nu_tv):
    """Re-estimate β and γ with time-varying ν held fixed."""
    
    T = len(S_data)
    y0 = [S_data[0], I_data[0], R_data[0]]

    def objective(params):
        beta, gamma = params
        if beta <= 0 or gamma <= 0:
            return 1e12
        sol = integrate(ode_lockdown_nu, y0, T, args=(beta, gamma, nu_tv, s_arr))
        return loss_sir(S_data, I_data, R_data, sol[:, 0], sol[:, 1], sol[:, 2])

    res = minimize(
        objective,
        x0=[0.4, 0.1],
        method="Nelder-Mead",
        options={"maxiter": 50_000, "xatol": 1e-10, "fatol": 1e-10, "adaptive": True},
    )

    beta, gamma = res.x
    sol = integrate(ode_lockdown_nu, y0, T, args=(beta, gamma, nu_tv, s_arr))

    l_sir = loss_sir(S_data, I_data, R_data, sol[:, 0], sol[:, 1], sol[:, 2])
    l_i = loss_i(I_data, sol[:, 1])

    Re_arr = (beta * (1 - s_arr) * sol[:, 0]) / gamma

    print(f"  beta     = {beta:.6f}   (paper India: 0.463)")
    print(f"  gamma    = {gamma:.6f}   (paper India: 0.114)")
    print(f"  R0 mean  = {np.mean(beta * (1 - s_arr) / gamma):.4f}")
    print(f"  Re mean  = {np.mean(Re_arr):.4f}")
    print(f"  Re range = [{Re_arr.min():.4f}, {Re_arr.max():.4f}]")
    print(f"  loss_SIR = {l_sir:,.3f}   (paper India: 29,116,762)")
    print(f"  loss_I   = {l_i:,.3f}   (paper India: 658,537)")

    return dict(beta=beta, gamma=gamma, sol=sol, Re=Re_arr, loss_sir=l_sir, loss_i=l_i)

def plot_all_models(us, results: dict, nu_tv: np.ndarray):
    dates = us["date"].values
    S_d = us["S_pct"].values
    I_d = us["I_pct"].values
    R_d = us["R_pct"].values

    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle("SIR Model Variants — United States", fontsize=15)

    model_keys = [
        ("simple", "Simple SIR"),
        ("lockdown", "SIR + Lockdown"),
        ("const_nu", "SIR + Lockdown + Constant ν"),
        ("final", "SIR + Lockdown + Time-varying ν"),
    ]

    for ax_row, (key, title) in zip(axes, model_keys):
        sol = results[key]["sol"]
        S_p = sol[:, 0] * 100
        I_p = sol[:, 1] * 100
        R_p = sol[:, 2] * 100

        # Left: all compartments
        ax = ax_row[0]
        ax.plot(dates, S_d, "b-", label="S (data)", lw=1.5)
        ax.plot(dates, I_d, "r-", label="I (data)", lw=1.5)
        ax.plot(dates, R_d, "g-", label="R (data)", lw=1.5)
        ax.plot(dates, S_p, "b--", label="S (model)", lw=1.2)
        ax.plot(dates, I_p, "r--", label="I (model)", lw=1.2)
        ax.plot(dates, R_p, "g--", label="R (model)", lw=1.2)
        ax.set_title(f"{title}\nloss_SIR={results[key]['loss_sir']:,.0f}")
        ax.set_ylabel("% of Population")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30)

        # Right: infected only (zoomed)
        ax2 = ax_row[1]
        ax2.plot(dates, I_d, "r-", label="I (data)", lw=1.5)
        ax2.plot(dates, I_p, "r--", label="I (model)", lw=1.2)
        ax2.set_title(f"Infected only\nloss_I={results[key]['loss_i']:,.0f}")
        ax2.set_ylabel("% of Population")
        ax2.legend(fontsize=8)
        ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(
        "./train/fitting/figures/sir_model_variants_US.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    print("  Saved: sir_model_variants_US.png")
    
results = {}
results["simple"] = fit_simple_sir(sir_data["S"].values, sir_data["I"].values, sir_data["R"].values)
results["lockdown"] = fit_sir_lockdown(
    sir_data["S"].values, sir_data["I"].values, sir_data["R"].values, sir_data["s_norm"].values
)
results["const_nu"] = fit_sir_lockdown_const_nu(
    sir_data["S"].values,
    sir_data["I"].values,
    sir_data["R"].values,
    sir_data["s_norm"].values
)
window_df, nu_arrays, nu_tv = window_search(
    results["const_nu"]["beta"],
    results["const_nu"]["gamma"],
    sir_data["S"].values,
    sir_data["I"].values,
    sir_data["R"].values,
    sir_data["s_norm"].values
)
results["final"] = fit_final_model(sir_data["S"].values, sir_data["I"].values, sir_data["R"].values, sir_data["s_norm"].values, nu_tv)
coeffs, poly_fn = fit_gdp_polynomial(sir_data)

plot_all_models(sir_data, results, nu_tv)
plot_window_search(window_df)
plot_nu_time_series(sir_data, nu_tv)
plot_gdp_stringency(sir_data, poly_fn)

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"{'Model':<35} {'loss_SIR':>15} {'loss_I':>13}")
print("-" * 55)
for key, label in [
    ("simple", "Simple SIR"),
    ("lockdown", "SIR + Lockdown"),
    ("const_nu", "SIR + Lockdown + Const ν"),
    ("final", "SIR + Lockdown + Time-varying ν"),
]:
    r = results[key]
    print(f"{label:<35} {r['loss_sir']:>15,.1f} {r['loss_i']:>13,.1f}")

params_out = {
    "beta_final": results["final"]["beta"],
    "gamma_final": results["final"]["gamma"],
    "nu_tv": nu_tv.tolist(),
    "s_arr": sir_data["s_norm"].values.tolist(),
    "gdp_coeffs": coeffs.tolist(),
}


with open("./train/sir_params_US.json", "w") as f:
    json.dump(params_out, f, indent=2)
print("\n  Parameters saved to sir_params_US.json")
print("  (Load these directly into your RL environment)")