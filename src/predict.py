import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from fetch_data import DataFetcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model_xgb.json"
FEATURES_PATH = MODEL_DIR / "model_features.pkl"
DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "forecast_results.json"
TZ = "Europe/Helsinki"


def apply_feature_engineering(df):
    """Replicate data engineering used in training"""
    df = df.copy()

    # Time features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Price lags
    df["price_lag_24"] = df["price"].shift(24)
    df["price_lag_168"] = df["price"].shift(168)

    # Domain features

    # Total cheap supply (nuclear + wind + solar)
    df["total_cheap_supply"] = (
        df["nuclear_mw"] + df["wind_forecast_72h_mw"] + df["solar_forecast_72h_mw"]
    )

    # Residual demand (consumption forecast - cheap supply)
    df["residual_demand"] = df["consumption_forecast_72h_mw"] - df["total_cheap_supply"]

    # Supply/demand ratio, (if ratio > 1 = cheap supply, if ratio < 1 = expensive supply)
    df["supply_demand_ratio"] = df["total_cheap_supply"] / (
        df["consumption_forecast_72h_mw"] + 1
    )

    # Rolling forecasts
    df["wind_forecast_roll_6h"] = df["wind_forecast_72h_mw"].rolling(window=6).mean()
    df["consumption_forecast_roll_6h"] = (
        df["consumption_forecast_72h_mw"].rolling(window=6).mean()
    )

    return df


def run_pipeline():
    print("Starting pipeline...")

    # Load raw data
    try:
        fetcher = DataFetcher()
        df = fetcher.run()
        print("Data fetching complete")

    except Exception as e:
        print(f"Fetching failed: {e}")
        sys.exit(1)

    # Apply features
    df = apply_feature_engineering(df)

    # Define history and prediction window
    now = pd.Timestamp.now(tz=TZ)

    # History: past 7 days
    start_history = now.normalize() - pd.Timedelta(days=7)
    X_hist = df.loc[start_history:now].copy()

    # Future
    end_horizon = now.normalize() + pd.Timedelta(days=2, hours=23)
    X_future = df.loc[now:end_horizon].copy()

    if X_future.empty:
        print("Error: No data found for future")
        return

    # Load model and features
    model = xgb.XGBRegressor()
    try:
        model.load_model(MODEL_PATH)
        with open(FEATURES_PATH, "rb") as f:
            feature_names = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return

    # Align features
    def prepare_data(df_slice):
        if df_slice.empty:
            return df_slice
        for col in feature_names:
            if col not in df_slice.columns:
                print(f"Warning: Missing column {col}, filling with 0")
                df_slice[col] = 0
        return df_slice[feature_names]

    X_hist_aligned = prepare_data(X_hist)
    X_future_aligned = prepare_data(X_future)

    # Predict
    hist_preds = model.predict(X_hist_aligned) if not X_hist_aligned.empty else []
    fut_preds = model.predict(X_future_aligned) if not X_future_aligned.empty else []

    # Format output
    output_data = {"history": [], "forecast": []}

    # Helper fuction to handle NaN values safely for JSON
    def safe_float(val):
        if pd.isna(val) or val is None:
            return None
        return float(val)

    # Fill history data
    for i, (timestamp, _) in enumerate(X_hist_aligned.iterrows()):
        real_price = df.loc[timestamp, "price"] if "price" in df.columns else None
        output_data["history"].append(
            {
                "timestamp": timestamp.isoformat(),
                "real_price": safe_float(real_price),
                "pred_price": safe_float(hist_preds[i]),
            }
        )

    # Fill Forecast data
    for i, (timestamp, _) in enumerate(X_future_aligned.iterrows()):
        row = df.loc[timestamp]
        real_price_future = row.get("price")

        output_data["forecast"].append(
            {
                "timestamp": timestamp.isoformat(),
                "pred_price": safe_float(fut_preds[i]),
                "real_price": safe_float(real_price_future),
                "wind_forecast": safe_float(row.get("wind_forecast_72h_mw")),
                "consumption_forecast": safe_float(
                    row.get("consumption_forecast_72h_mw")
                ),
                "temp_helsinki": safe_float(row.get("temp_helsinki")),
                "residual_demand": safe_float(row.get("residual_demand")),
            }
        )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()
