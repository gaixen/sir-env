import pandas as pd
from scipy.optimize import minimize
import logging
import os
from train.fitting.commons import (
    loss_sir,
    loss_i,
    ode_simple,
    integrate
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

logger.info("\nCompartment sanity check (first non-zero I row):")
first = sir_data[sir_data["I"] > 0].iloc[0]
logger.info(
    f"  date={first['date'].date()}, S={first['S']:.4f}, "
    f"I={first['I']:.6f}, R={first['R']:.4f}"
)

def fit_simple_sir(S_data, I_data, R_data):
    print("\n" + "=" * 55)
    print("MODEL 2a: Simple SIR")
    print("=" * 55)

    T = len(S_data)
    y0 = [S_data[0], I_data[0], R_data[0]]

    def objective(params):
        beta, gamma = params
        if beta <= 0 or gamma <= 0:
            return 1e12
        sol = integrate(ode_simple, y0, T, args=(beta, gamma))
        return loss_sir(S_data, I_data, R_data, sol[:, 0], sol[:, 1], sol[:, 2])

    res = minimize(
        objective,
        x0=[0.1, 0.05],
        method="Nelder-Mead",
        options={"maxiter": 50_000, "xatol": 1e-10, "fatol": 1e-10, "adaptive": True},
    )

    beta, gamma = res.x
    sol = integrate(ode_simple, y0, T, args=(beta, gamma))

    l_sir = loss_sir(S_data, I_data, R_data, sol[:, 0], sol[:, 1], sol[:, 2])
    l_i = loss_i(I_data, sol[:, 1])

    print(f"  beta  = {beta:.6f}   (paper: 0.042)")
    print(f"  gamma = {gamma:.6f}   (paper: 0.024)")
    print(f"  R0    = {beta / gamma:.4f}   (paper: 1.762)")
    print(f"  loss_SIR = {l_sir:,.3f}   (paper: 85,051,490)")
    print(f"  loss_I   = {l_i:,.3f}   (paper: 45,187,665)")

    return dict(beta=beta, gamma=gamma, sol=sol, loss_sir=l_sir, loss_i=l_i)
