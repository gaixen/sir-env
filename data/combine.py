import pandas as pd


us_cases_death = pd.read_csv(r'data\deaths\us_cases_deaths_interpolated.csv')
us_gdp_daily = pd.read_csv(r'data\gdp\gdp_us_daily.csv')
us_policy_response_data = pd.read_csv(r'C:\Users\DELL\scaler\data\policy\policy_response_data.csv')
us_vaccination_data = pd.read_csv(r'data\vaccinations\vaccinations_us_interpolated.csv')

_dataframes = [us_cases_death, us_gdp_daily, us_policy_response_data, us_vaccination_data]
_start_time = max(us_cases_death.date.min(), us_gdp_daily.date.min(), us_policy_response_data.date.min(), us_vaccination_data.date.min())
# 2020-12-20
_end_time = min(us_cases_death.date.max(), us_gdp_daily.date.max(), us_policy_response_data.date.max(), us_vaccination_data.date.max())
# 2022-12-31
us_cases_death = us_cases_death.drop(columns= ['country'])
us_policy_response_data = us_policy_response_data.drop(columns= ['country'])
us_vaccination_data = us_vaccination_data.drop(columns= ['state'])
# clip between two points
processed = []
for df in _dataframes:
    df = df.copy()
    if "date" not in df.columns:
        raise KeyError("expected column 'date' in all input dataframes")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    mask = (df["date"] >= pd.to_datetime(_start_time)) & (df["date"] <= pd.to_datetime(_end_time))
    df = df.loc[mask]
    df = df.set_index("date")
    processed.append(df)

combined = pd.concat(processed, axis=1, join="inner")
combined = combined.drop(columns = ["state", 
                                    "total_deaths_last12m", 
                                    "total_deaths_per_100k_last12m", 
                                    "total_deaths_per_million_last12m"])
combined = combined.dropna(subset = ["total_cases"])
combined = combined.reset_index()

combined_path = r"data/combined_us_data.csv"
combined.to_csv(combined_path, index=False)
print(f"Wrote combined dataset with shape {combined.shape} to {combined_path}")

