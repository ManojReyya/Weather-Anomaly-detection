"""
etrain.py - Generate synthetic weather dataset, train RandomForest model, save artifacts.

Outputs:
- data/weather_synthetic.csv: synthetic dataset
- models/weather_rf.pkl: trained RandomForestClassifier model
- models/metadata.json: feature metadata for app usage

Run:
python etrain.py
"""

import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATA_DIR = os.path.join("data")
MODELS_DIR = os.path.join("models")
DATA_PATH = os.path.join(DATA_DIR, "weather_synthetic.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "weather_rf.pkl")
META_PATH = os.path.join(MODELS_DIR, "metadata.json")

np.random.seed(42)

@dataclass
class FeatureConfig:
    # Value ranges used both for synthetic data and UI constraints
    temperature_c: Tuple[float, float] = (-40.0, 50.0)
    humidity_pct: Tuple[float, float] = (0.0, 100.0)
    wind_speed_ms: Tuple[float, float] = (0.0, 60.0)
    precipitation_mm: Tuple[float, float] = (0.0, 300.0)
    pressure_hpa: Tuple[float, float] = (870.0, 1050.0)
    cloud_cover_pct: Tuple[float, float] = (0.0, 100.0)

FEATURES = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "precipitation_mm",
    "pressure_hpa",
    "cloud_cover_pct",
]

LABEL = "risk_level"  # 0=Normal, 1=Elevated, 2=High (extreme)


def synthesize(n_samples: int = 8000, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    # Base distributions
    temp = np.random.normal(loc=18, scale=12, size=n_samples)
    hum = np.clip(np.random.normal(loc=60, scale=20, size=n_samples), 0, 100)
    wind = np.abs(np.random.normal(loc=6, scale=6, size=n_samples))
    precip = np.abs(np.random.gamma(shape=2.0, scale=5.0, size=n_samples))
    press = np.random.normal(loc=1013, scale=15, size=n_samples)
    cloud = np.clip(np.random.normal(loc=50, scale=35, size=n_samples), 0, 100)

    # Clip to realistic UI ranges
    temp = np.clip(temp, *cfg.temperature_c)
    wind = np.clip(wind, *cfg.wind_speed_ms)
    precip = np.clip(precip, *cfg.precipitation_mm)
    press = np.clip(press, *cfg.pressure_hpa)

    df = pd.DataFrame({
        "temperature_c": temp,
        "humidity_pct": hum,
        "wind_speed_ms": wind,
        "precipitation_mm": precip,
        "pressure_hpa": press,
        "cloud_cover_pct": cloud,
    })

    # Heuristic risk rules to generate labels
    risk = np.zeros(n_samples, dtype=int)

    # High risk when multiple extremes coincide
    high_mask = (
        (df["wind_speed_ms"] > 25) & (df["precipitation_mm"] > 60)
    ) | (
        (df["temperature_c"] > 40) & (df["humidity_pct"] < 15)
    ) | (
        (df["temperature_c"] < -20) & (df["wind_speed_ms"] > 15)
    ) | (
        (df["pressure_hpa"] < 890) & (df["wind_speed_ms"] > 30)
    )

    risk[high_mask] = 2

    # Elevated risk conditions
    elev_mask = (
        (df["wind_speed_ms"].between(15, 25))
        | (df["precipitation_mm"].between(30, 60))
        | (df["temperature_c"].between(35, 40))
        | (df["temperature_c"].between(-10, -20))
        | (df["humidity_pct"] < 20)
        | (df["pressure_hpa"] < 950)
        | (df["cloud_cover_pct"] > 85)
    ) & (risk < 2)

    risk[elev_mask] = 1

    df[LABEL] = risk
    return df


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    X = df[FEATURES]
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    print("\nClassification report:\n", report)

    return clf


def save_artifacts(df: pd.DataFrame, model: RandomForestClassifier):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df.to_csv(DATA_PATH, index=False)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "features": FEATURES,
        "label": LABEL,
        "classes": {
            "0": "Normal",
            "1": "Elevated",
            "2": "High",
        },
        "ranges": {
            "temperature_c": [-40.0, 50.0],
            "humidity_pct": [0.0, 100.0],
            "wind_speed_ms": [0.0, 60.0],
            "precipitation_mm": [0.0, 300.0],
            "pressure_hpa": [870.0, 1050.0],
            "cloud_cover_pct": [0.0, 100.0],
        },
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    df = synthesize(n_samples=8000)
    model = train_model(df)
    save_artifacts(df, model)
    print(f"Saved dataset to {DATA_PATH}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {META_PATH}")