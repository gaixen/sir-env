import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

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


def fit_gdp_polynomial(us: pd.DataFrame):
    print("\n" + "=" * 55)
    print("GDP–Stringency Polynomial (degree 3)")
    print("=" * 55)

    sub = us[["stringency_index", "GDP_scaled"]].dropna()
    s_vals = sub["stringency_index"].values
    gdp_vals = sub["GDP_scaled"].values

    coeffs = np.polyfit(s_vals, gdp_vals, deg=3)
    p = np.poly1d(coeffs)

    gdp_pred = p(s_vals)
    ss_res = np.sum((gdp_vals - gdp_pred) ** 2)
    ss_tot = np.sum((gdp_vals - gdp_vals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    corr = np.corrcoef(s_vals, gdp_vals)[0, 1]

    from scipy.stats import pearsonr

    _, pval = pearsonr(s_vals, gdp_vals)

    print(f"  Coefficients (a,b,c,d): {coeffs}")
    print(f"  Pearson r  = {corr:.5f}")
    print(f"  R²         = {r2:.5f}")
    print(f"  p-value    = {pval:.6f}")

    return coeffs, p


def plot_gdp_stringency(us, poly_fn):
    s_range = np.linspace(
        us["stringency_index"].min(), us["stringency_index"].max(), 200
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stringency vs GDP — United States")

    axes[0].scatter(
        us["stringency_index"], us["GDP_scaled"], s=10, alpha=0.5, label="data"
    )
    axes[0].plot(s_range, poly_fn(s_range), "r-", lw=2, label="Degree-3 fit")
    axes[0].set_xlabel("Stringency Index")
    axes[0].set_ylabel("GDP_scaled")
    axes[0].legend()

    axes[1].plot(us["date"], us["GDP_scaled"], "g-", label="Actual GDP_scaled")
    axes[1].plot(
        us["date"],
        poly_fn(us["stringency_index"].fillna(0)),
        "g--",
        label="Modelled GDP_scaled",
    )
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(
        "./train/fitting/figures/gdp_stringency_US.png", dpi=150, bbox_inches="tight"
    )
    plt.show()