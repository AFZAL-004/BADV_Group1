
"""
Exact reusable preprocessing for Airline Route Profitability Streamlit prediction.

This mirrors the uploaded preprocessing notebook:
01_Group_Data_Understanding_Preprocessing_EDA.ipynb

Important:
- Output is the 37 unscaled feature columns used by X_train.csv / X_test.csv.
- The dashboard should apply processed_data/scaler.pkl only for SVM, LR+Polynomial, and MLP.
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


TARGET_COL = "Profitable"

LEAKY_COLS = [
    "Profit", "Profit_Margin",
    "Total_Revenue", "Total_Cost",
    "Ticket_Revenue", "Ancillary_Revenue", "Cargo_Revenue",
    "Fuel_Cost", "Maintenance_Cost", "Crew_Cost", "Depreciation_Cost",
    "Insurance_Cost", "Airport_Fees", "Catering_Cost", "Handling_Cost",
    "Navigation_Fees", "Sales_Distribution_Cost", "Passenger_Service_Cost",
    "Overhead_Cost", "Marketing_Cost", "IT_Systems_Cost",
]

ID_COLS = ["Flight_ID", "Flight_Number", "Route"]

RESIDUAL_STRING_COLS = [
    "Flight_Date", "Day_of_Week", "Airline", "Origin", "Destination",
    "Origin_Country", "Destination_Country", "Aircraft_Type", "Season", "Demand_Level",
]

REGION_MAP = {
    "USA": "North_America", "Canada": "North_America", "Mexico": "North_America",
    "UK": "Europe", "France": "Europe", "Germany": "Europe", "Spain": "Europe",
    "Italy": "Europe", "Netherlands": "Europe", "Switzerland": "Europe",
    "Turkey": "Europe", "Sweden": "Europe", "Norway": "Europe",
    "China": "Asia", "Japan": "Asia", "India": "Asia", "South Korea": "Asia",
    "Singapore": "Asia", "Hong Kong": "Asia", "Thailand": "Asia", "Malaysia": "Asia",
    "UAE": "Middle_East", "Saudi Arabia": "Middle_East", "Qatar": "Middle_East",
    "Brazil": "South_America", "Argentina": "South_America", "Colombia": "South_America",
    "South Africa": "Africa", "Nigeria": "Africa", "Kenya": "Africa", "Egypt": "Africa",
    "Australia": "Oceania", "New Zealand": "Oceania",
}

SEASON_MAP = {"Low": 0, "Normal": 1, "Shoulder": 2, "Peak": 3}
DEMAND_MAP = {"Low": 0, "Medium": 1, "High": 2}

# Exact IQR bounds printed in the preprocessing notebook.
WINSOR_BOUNDS = {
    "Flight_Hours": (-7.55, 22.45),
    "Competition_Index": (-3.00, 13.00),
    "Aircraft_Age_Years": (-9.00, 31.00),
    "Passengers": (34.50, 422.50),
}

LOG1P_COLS = ["Flight_Distance_KM", "Average_Ticket_Price", "Delay_Minutes"]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ensure_numeric(df: pd.DataFrame, col: str, fill_value: float = 0.0) -> None:
    if col in df.columns:
        df[col] = _to_num(df[col]).fillna(fill_value)


def preprocess_raw_route_row(
    raw_row: pd.DataFrame,
    feature_cols: list[str],
    train_medians: Dict[str, float],
    stats: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Convert one original raw dataset row to the 37 unscaled processed features.

    This follows the preprocessing notebook order:
    1. drop leaky/id columns
    2. constraint fixes
    3. winsorisation
    4. log1p transforms
    5. feature engineering
    6. country-to-region mapping
    7. ordinal encoding
    8. one-hot encoding with drop_first=True
    9. drop residual string columns
    10. align to feature_names.json
    """
    if raw_row.empty:
        raise ValueError("raw_row is empty")

    df = raw_row.copy().reset_index(drop=True)

    # Drop leakage, ids, and target from raw input before inference.
    df.drop(columns=[c for c in LEAKY_COLS + ID_COLS + [TARGET_COL] if c in df.columns], inplace=True, errors="ignore")

    # Numeric conversion for known numeric fields.
    known_numeric = [
        "Flight_Distance_KM", "Aircraft_Capacity", "Aircraft_Age_Years", "Passengers",
        "Load_Factor", "Flight_Hours", "Competition_Index", "Weather_Disruption",
        "On_Time_Performance", "Delay_Minutes", "Average_Ticket_Price",
        "Fuel_Price_Per_Liter", "Passenger_Satisfaction_Score", "Market_Share_Pct",
    ]
    for col in known_numeric:
        _ensure_numeric(df, col, train_medians.get(col, 0.0))

    # Dirty data constraint fixes from the notebook.
    if "Load_Factor" in df.columns:
        df["Load_Factor"] = df["Load_Factor"].clip(upper=1.0)
    if "Delay_Minutes" in df.columns:
        df["Delay_Minutes"] = df["Delay_Minutes"].clip(lower=0)
    if "Average_Ticket_Price" in df.columns:
        df["Average_Ticket_Price"] = df["Average_Ticket_Price"].clip(lower=30)

    # IQR winsorisation from the notebook.
    for col, (lower, upper) in WINSOR_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)

    # log1p transforms from the notebook.
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Feature engineering from the notebook.
    if "Average_Ticket_Price" in df.columns and "Flight_Distance_KM" in df.columns:
        df["price_per_km"] = df["Average_Ticket_Price"] / (df["Flight_Distance_KM"] + 1)

    if "Delay_Minutes" in df.columns:
        df["delay_flag"] = (df["Delay_Minutes"] > 30).astype(int)

    if "Flight_Date" in df.columns:
        flight_date = pd.to_datetime(df["Flight_Date"], errors="coerce")
        df["flight_month"] = flight_date.dt.month.fillna(train_medians.get("flight_month", 1)).astype(int)
        df["is_weekend"] = (flight_date.dt.dayofweek >= 5).fillna(False).astype(int)

    narrow_body = ["737", "A320", "A321"]
    if "Aircraft_Type" in df.columns:
        df["is_narrow_body"] = df["Aircraft_Type"].apply(
            lambda x: 1 if any(nb in str(x) for nb in narrow_body) else 0
        )
    else:
        df["is_narrow_body"] = 0

    # Country-to-region mapping from notebook.
    if "Origin_Country" in df.columns:
        df["origin_region"] = df["Origin_Country"].map(REGION_MAP).fillna("Other")
    if "Destination_Country" in df.columns:
        df["dest_region"] = df["Destination_Country"].map(REGION_MAP).fillna("Other")

    # Ordinal encodings from notebook.
    if "Season" in df.columns:
        df["Season_Ordinal"] = df["Season"].map(SEASON_MAP).fillna(train_medians.get("Season_Ordinal", 1)).astype(int)

    if "Demand_Level" in df.columns:
        df["Demand_Ordinal"] = df["Demand_Level"].map(DEMAND_MAP).fillna(train_medians.get("Demand_Ordinal", 1)).astype(int)

    # One-hot encoding from notebook, exactly these columns and drop_first=True.
    ohe_cols = [c for c in ["Route_Category", "Alliance", "origin_region", "dest_region"] if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Drop residual string columns from notebook.
    df.drop(columns=[c for c in RESIDUAL_STRING_COLS if c in df.columns], inplace=True, errors="ignore")

    # Convert bool one-hot columns to 0/1.
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Align exactly to trained features.
    processed = pd.DataFrame(index=[0], columns=feature_cols)
    for col in feature_cols:
        if col in df.columns:
            processed.loc[0, col] = df.loc[0, col]
        else:
            processed.loc[0, col] = train_medians.get(col, 0.0)

    processed = processed.apply(pd.to_numeric, errors="coerce")
    for col in processed.columns:
        processed[col] = processed[col].fillna(train_medians.get(col, 0.0))

    return processed[feature_cols]


def make_model_input(
    processed_unscaled: pd.DataFrame,
    needs_scaled: bool,
    scaler: Any | None,
) -> pd.DataFrame:
    """Apply model-specific scaling exactly at the final input stage."""
    if needs_scaled:
        if scaler is None:
            raise FileNotFoundError("processed_data/scaler.pkl was not found")
        return pd.DataFrame(
            scaler.transform(processed_unscaled),
            columns=processed_unscaled.columns,
            index=processed_unscaled.index,
        )
    return processed_unscaled.copy()
