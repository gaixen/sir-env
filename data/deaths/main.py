import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datetime import datetime
from owid.catalog import fetch

warnings.filterwarnings("ignore")
# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
_PATH = os.path.dirname(os.path.abspath(__file__))


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


deaths_df = fetch("garden/covid/latest/cases_deaths/cases_deaths")

_countries = deaths_df.index.get_level_values("country").unique()
_features = deaths_df.columns
us_cases_deaths = deaths_df.loc[["United States"]]
_column_entries = us_cases_deaths.isnull()
logger.info(f"Missing values in US cases and deaths: {_column_entries}")
_time_span = get_time_span(us_cases_deaths)

_columns_to_drop_nulls_from = [
    "new_deaths",
]


def drop_nulls_compat(df, subset):
    if hasattr(df, "drop_nulls"):
        return df.drop_nulls(subset=subset)
    if hasattr(df, "dropna"):
        return df.dropna(subset=subset)
    return pd.DataFrame(df).dropna(subset=subset)


us_cases_deaths = drop_nulls_compat(us_cases_deaths, _columns_to_drop_nulls_from)
logger.info(f"Shape after dropping nulls: {getattr(us_cases_deaths, 'shape', None)}")


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

    # Case 1: MultiIndex with a single country (common in this dataset)
    if isinstance(df_copy.index, pd.MultiIndex):
        names = list(df_copy.index.names)
        if "date" in names:
            country_level = [n for n in names if n != "date"]
            if len(country_level) != 1:
                raise ValueError(
                    "Expected a single non-date level (e.g. 'country') in the MultiIndex"
                )
            country_name = country_level[0]
            countries = df_copy.index.get_level_values(country_name).unique()
            if len(countries) != 1:
                raise ValueError(
                    "DataFrame contains multiple entities; pass a single-entity DataFrame"
                )
            country_val = countries[0]

            # Build a full MultiIndex for the requested date range and reindex
            new_index = pd.MultiIndex.from_product(
                [[country_val], date_range], names=[country_name, "date"]
            )
            df_reindexed = df_copy.reindex(new_index)

            # Extract series for interpolation (dropping the country level)
            ser = df_reindexed[col].droplevel(country_name)
            ser.index = pd.to_datetime(ser.index)
            ser = ser.reindex(date_range)
            ser = ser.sort_index().interpolate(method="time", limit_direction="both")
            ser = ser.fillna(method="ffill").fillna(method="bfill")

            # Put back into DataFrame
            df_reindexed[col] = pd.Series(ser.values, index=new_index)
            return df_reindexed
        else:
            raise ValueError("MultiIndex provided but no 'date' level found")

    # Case 2: DatetimeIndex or plain index with date-like values
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


us_cases_deaths_interpolated = interpolate_missing_values(
    us_cases_deaths,
    "new_cases",
    get_time_span(us_cases_deaths, "date")[0],
    get_time_span(us_cases_deaths, "date")[1],
)
# Save plots to a `figures` folder under the script directory (do not hardcode paths)
FIGURES_DIR = os.path.join(_PATH, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


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


us_cases_deaths_interpolated = to_pandas_compat(us_cases_deaths_interpolated)
# Create the plots
plt.figure(figsize=(15, 10))

# Plotting New Cases against Time
plt.subplot(3, 1, 1)  # 3 rows, 1 column, first plot
sns.lineplot(
    data=us_cases_deaths_interpolated,
    x="date",
    y="new_cases",
    color="skyblue",
)
plt.title("New COVID-19 Cases in the US Over Time")
plt.xlabel("Date")
plt.ylabel("New Cases")
plt.grid(True)

plt.subplot(3, 1, 2)  # 3 rows, 1 column, second plot
sns.lineplot(
    data=us_cases_deaths_interpolated,
    x="date",
    y="total_cases",
    color="salmon",
)
plt.title("Total COVID-19 Deaths in the US Over Time")
plt.xlabel("Date")
plt.ylabel("Total Deaths")
plt.grid(True)

plt.subplot(3, 1, 3)  # 3 rows, 1 column, third plot
sns.lineplot(
    data=us_cases_deaths_interpolated,
    x="date",
    y="new_deaths",
    color="salmon",
)
plt.title("New Deaths in the US Over Time")
plt.xlabel("Date")
plt.ylabel("New Deaths")
plt.grid(True)

plt.tight_layout()
combined_path = os.path.join(FIGURES_DIR, "covid_plots.png")
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
logger.info(f"Saved combined plot to {combined_path}")
plt.show()
