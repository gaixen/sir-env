import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
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


policy_response_data = fetch("garden/covid/latest/oxcgrt_policy/oxcgrt_policy")
policy_response_data = to_pandas_compat(policy_response_data)
policy_response_data = policy_response_data[
    policy_response_data.index.isin(["United States"], level="country")
]
logger.info(f"Policy response data shape: {policy_response_data.shape}")
_features = policy_response_data.columns
logger.info(f"features: {_features}")
_time_span = get_time_span(policy_response_data)
logger.info(f"Time span: {_time_span}")
_columns_to_drop = [
    "e3_fiscal_measures",
    "e4_international_support",
    "h4_emergency_investment_in_healthcare",
    "v2b_vaccine_age_eligibility_availability_age_floor__general_population_summary",
    "v2c_vaccine_age_eligibility_availability_age_floor__at_risk_summary",
    "v2_pregnant_people",
    "stringency_index_weighted_average",
    "h5_investment_in_vaccines",
]

policy_response_data = policy_response_data.drop(columns=_columns_to_drop)
policy_response_data.to_csv("./data/policy/policy_response_data.csv", index=True)
