import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datetime import datetime
from owid.catalog import fetch

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
_PATH = os.path.dirname(os.path.abspath(__file__))


def to_pandas_compat(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    try:
        import polars as _pl

        if isinstance(obj, _pl.DataFrame):
            return obj.to_pandas()
    except Exception:
        pass

    if hasattr(obj, "to_pandas") and callable(obj.to_pandas):
        return obj.to_pandas()
    if hasattr(obj, "to_df") and callable(obj.to_df):
        return obj.to_df()
    return pd.DataFrame(obj)


def get_time_span(obj, date_level="date"):
    if hasattr(obj, "index"):
        index = obj.index
    else:
        index = obj
    if isinstance(index, pd.MultiIndex):
        if date_level in index.names:
            date_vals = index.get_level_values(date_level)
        else:
            date_vals = index.get_level_values(1)
    else:
        date_vals = index

    date_vals = pd.to_datetime(date_vals)
    return date_vals.min(), date_vals.max()


vaccinations_us = fetch("garden/covid/latest/vaccinations_us/vaccinations_us")
_features = vaccinations_us.columns
logger.debug(f"Features: {_features}")
vaccinations_us = to_pandas_compat(vaccinations_us)
_time_span = get_time_span(vaccinations_us, date_level="date")

logger.debug(f"Time span: {_time_span}")


def interpolate_missing_values(
    df: pd.DataFrame,
    col: str,
    start: datetime = _time_span[0],
    end: datetime = _time_span[1],
) -> pd.DataFrame:

    df_copy = df.copy()

    if col not in df_copy.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    date_range = pd.date_range(start=start, end=end, freq="D")

    if isinstance(df_copy.index, pd.MultiIndex):
        names = list(df_copy.index.names)
        if "date" in names:
            group_levels = [n for n in names if n != "date"]
            if len(group_levels) == 0:
                pass
            else:
                df_reset = df_copy.reset_index()
                df_reset["date"] = pd.to_datetime(df_reset["date"])
                groups = []
                for key, grp in df_reset.groupby(group_levels):
                    grp = grp.set_index("date").sort_index()
                    grp = grp.reindex(date_range)
                    grp.index.name = "date"

                    # Interpolate the target column
                    if col not in grp.columns:
                        raise KeyError(f"Column '{col}' not found for interpolation")
                    ser = (
                        grp[col]
                        .sort_index()
                        .interpolate(method="time", limit_direction="both")
                    )
                    ser = ser.fillna(method="ffill").fillna(method="bfill")
                    grp[col] = ser.values

                    if isinstance(key, tuple):
                        for lvl, val in zip(group_levels, key):
                            grp[lvl] = val
                    else:
                        grp[group_levels[0]] = key

                    grp = grp.reset_index()
                    index_keys = group_levels + ["date"]
                    grp = grp.set_index(index_keys)
                    groups.append(grp)

                if len(groups) == 0:
                    return df_copy
                df_out = pd.concat(groups)
                return df_out.reindex(columns=df_copy.columns)
        else:
            raise ValueError("MultiIndex provided but no 'date' level found")

    idx = pd.to_datetime(df_copy.index)
    df_copy.index = idx
    df_reindexed = df_copy.reindex(date_range)
    ser = (
        df_reindexed[col]
        .sort_index()
        .interpolate(method="time", limit_direction="both")
    )
    ser = ser.fillna(method="ffill").fillna(method="bfill")
    df_reindexed[col] = ser.values
    return df_reindexed


vaccinations_us_interpolated = interpolate_missing_values(
    vaccinations_us, "daily_vaccinations", start=_time_span[0], end=_time_span[1]
)

plt.figure(figsize=(15, 10))
plot_df = vaccinations_us_interpolated.reset_index()
plot_df["date"] = pd.to_datetime(plot_df["date"])
daily_plot_df = plot_df.groupby("date", as_index=False).sum()
daily_plot_df.to_csv(
    "./data/vaccinations/vaccinations_us_interpolated.csv", index=False
)
sns.lineplot(
    data=daily_plot_df,
    x="date",
    y="daily_vaccinations",
    color="skyblue",
)
plt.title("daily_vaccinations")
plt.xlabel("Date")
plt.ylabel("daily_vaccinations")
plt.grid(True)

FIGURES_DIR = os.path.join(_PATH, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
combined_path = os.path.join(FIGURES_DIR, "Vaccinations_US.png")
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
logger.info(f"Saved combined plot to {combined_path}")
