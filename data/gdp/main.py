import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
_PATH = os.path.dirname(os.path.abspath(__file__))

gdp_us_data = pd.read_csv(r"data\policy\GDP.csv")


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


_features = gdp_us_data.columns
logger.debug(f"Features: {_features}")
_time_span = get_time_span(gdp_us_data, "observation_date")


def interpolate_to_daily(
    df: pd.DataFrame,
    date_col: str,
    col: str,
    method: str = "polynomial",
    order: int = 2,
) -> pd.DataFrame:

    dfc = df.copy()
    if date_col not in dfc.columns:
        raise KeyError(f"Date column '{date_col}' not found")
    dfc[date_col] = pd.to_datetime(dfc[date_col])
    dfc = dfc.set_index(date_col).sort_index()

    start, end = dfc.index.min(), dfc.index.max()
    full_idx = pd.date_range(start=start, end=end, freq="D")
    df_reindexed = dfc.reindex(full_idx)

    if col not in df_reindexed.columns:
        raise KeyError(f"Column '{col}' not found for interpolation")

    ser = df_reindexed[col].astype(float)
    if method == "polynomial":
        mask = ser.dropna().index
        if len(mask) < order + 1:
            ser_interp = ser.interpolate(method="time", limit_direction="both")
        else:
            x = (mask.astype("int64") // 10**9).astype(float)
            y = ser.loc[mask].values.astype(float)
            coeffs = np.polyfit(x, y, order)
            x_full = (ser.index.astype("int64") // 10**9).astype(float)
            ser_interp = pd.Series(np.polyval(coeffs, x_full), index=ser.index)
            ser_interp.loc[mask] = ser.loc[mask]
    else:
        ser_interp = ser.interpolate(method=method, limit_direction="both")

    ser_interp = ser_interp.fillna(method="ffill").fillna(method="bfill")
    df_reindexed[col] = ser_interp.values
    return df_reindexed


def minmax_normalize_series(ser: pd.Series, feature_range=(85, 100)) -> pd.Series:
    scaler = MinMaxScaler(feature_range=feature_range)
    arr = ser.values.reshape(-1, 1)
    scaled = scaler.fit_transform(arr)
    return pd.Series(scaled.ravel(), index=ser.index)


target_col = _features[1]
gdp_daily = interpolate_to_daily(
    gdp_us_data, "observation_date", target_col, method="polynomial", order=2
)
gdp_daily[f"{target_col}_scaled"] = minmax_normalize_series(
    gdp_daily[target_col], feature_range=(85, 100)
)
gdp_daily.to_csv("./data/policy/gdp_us_daily.csv")
logger.info(f"Interpolated {target_col} to daily frequency and added scaled column")

plt.figure(figsize=(15, 10))
sns.lineplot(
    data=gdp_daily,
    x=gdp_daily.index,
    y="GDP",
    color="skyblue",
)
plt.title("GDP US Daily")
plt.xlabel("Date")
plt.ylabel("GDP(in Billion dollars)")
plt.grid(True)

FIGURES_DIR = os.path.join(_PATH, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
combined_path = os.path.join(FIGURES_DIR, "GDP_US_daily.png")
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
logger.info(f"Saved combined plot to {combined_path}")
