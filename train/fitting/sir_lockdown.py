import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging
import os
import matplotlib.pyplot as plt
from train.fitting.commons import (
    loss_sir,
    loss_i,
    ode_lockdown,
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


def fit_sir_lockdown(S_data, I_data, R_data, s_arr):
    print("\n" + "=" * 55)
    print("MODEL 2b: SIR + Lockdown")
    print("=" * 55)

    T = len(S_data)
    y0 = [S_data[0], I_data[0], R_data[0]]

    def objective(params):
        beta, gamma = params
        if beta <= 0 or gamma <= 0:
            return 1e12
        sol = integrate(ode_lockdown, y0, T, args=(beta, gamma, s_arr))
        return loss_sir(S_data, I_data, R_data, sol[:, 0], sol[:, 1], sol[:, 2])

    res = minimize(
        objective,
        x0=[0.4, 0.09],
        method="Nelder-Mead",
        options={"maxiter": 50_000, "xatol": 1e-10, "fatol": 1e-10, "adaptive": True},
    )

    beta, gamma = res.x
    sol = integrate(ode_lockdown, y0, T, args=(beta, gamma, s_arr))

    l_sir = loss_sir(S_data, I_data, R_data, sol[:, 0], sol[:, 1], sol[:, 2])
    l_i = loss_i(I_data, sol[:, 1])

    Re_arr = beta * (1 - s_arr) * S_data / 1.0  # Re(t) = β(t)·S(t)/γ...
    Re_arr = Re_arr / gamma  # corrected
    print(f"  beta   = {beta:.6f}   (paper: 0.401)")
    print(f"  gamma  = {gamma:.6f}   (paper: 0.090)")
    print(f"  R0 mean= {np.mean(beta * (1 - s_arr) / gamma):.4f}")
    print(f"  loss_SIR = {l_sir:,.3f}   (paper: 98,438,821)")
    print(f"  loss_I   = {l_i:,.3f}   (paper: 11,345,389)")

    return dict(beta=beta, gamma=gamma, sol=sol, loss_sir=l_sir, loss_i=l_i)


def fit_sir_lockdown_const_nu(S_data, I_data, R_data, s_arr):
    print("\n" + "=" * 55)
    print("MODEL 2c: SIR + Lockdown + Constant ν")
    print("=" * 55)

    T = len(S_data)
    y0 = [S_data[0], I_data[0], R_data[0]]

    def objective(params):
        beta, gamma, nu = params
        if beta <= 0 or gamma <= 0 or nu < 0:
            return 1e12
        sol = integrate(ode_lockdown_nu, y0, T, args=(beta, gamma, nu, s_arr))
        return loss_sir(S_data, I_data, R_data, sol[:, 0], sol[:, 1], sol[:, 2])

    res = minimize(
        objective,
        x0=[0.4, 0.09, 1e-5],
        method="Nelder-Mead",
        options={"maxiter": 50_000, "xatol": 1e-10, "fatol": 1e-10, "adaptive": True},
    )

    beta, gamma, nu = res.x
    nu = max(nu, 0.0)
    sol = integrate(ode_lockdown_nu, y0, T, args=(beta, gamma, nu, s_arr))

    l_sir = loss_sir(S_data, I_data, R_data, sol[:, 0], sol[:, 1], sol[:, 2])
    l_i = loss_i(I_data, sol[:, 1])

    print(f"  beta  = {beta:.6f}   (paper: 0.409)")
    print(f"  gamma = {gamma:.6f}   (paper: 0.092)")
    print(f"  nu    = {nu:.2e}   (paper: 2.9e-05)")
    print(f"  loss_SIR = {l_sir:,.3f}   (paper: 94,636,860)")
    print(f"  loss_I   = {l_i:,.3f}   (paper: 10,840,360)")

    return dict(beta=beta, gamma=gamma, nu=nu, sol=sol, loss_sir=l_sir, loss_i=l_i)


def plot_nu_time_series(us, nu_tv):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(us["date"].values, nu_tv, color="orange", lw=1.5)
    ax.set_title("Time-varying Vaccination Rate ν(t) — United States")
    ax.set_ylabel("ν (daily fraction of population)")
    ax.set_xlabel("Date")

    # Mark vaccine rollout — US started Dec 14, 2020
    vax_date = pd.Timestamp("2020-12-14")
    if vax_date >= us["date"].min():
        ax.axvline(
            vax_date, color="red", ls="--", label="Vaccine rollout (Dec 14, 2020)"
        )
        ax.legend()

    plt.tight_layout()
    # Image saving disabled to prevent writing files during automated runs
    logger.info("Skipping saving figure: nu_timeseries_US.png")
    plt.show()