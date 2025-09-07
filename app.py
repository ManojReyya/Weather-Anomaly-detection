"""
app.py - CLI for Weather Anomaly Detection

- Loads RandomForest model and metadata produced by train.py
- Accepts weather features via command-line flags or interactive prompts
- Predicts risk level and prints safety recommendations

Usage examples:
1) Provide all features via flags:
   python app.py --temperature_c 28 --humidity_pct 70 --wind_speed_ms 5 \
                 --precipitation_mm 2 --pressure_hpa 1012 --cloud_cover_pct 40

2) Interactive (missing values will be prompted):
   python app.py --temperature_c 30 --humidity_pct 50

3) JSON output for programmatic use:
   python app.py --temperature_c 28 --humidity_pct 70 --wind_speed_ms 5 \
                 --precipitation_mm 2 --pressure_hpa 1012 --cloud_cover_pct 40 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "weather_rf.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "metadata.json")


def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        raise FileNotFoundError(
            "Model or metadata not found. Please run 'python train.py' first to generate them."
        )
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def recommend_safety(risk_label: int, features: Dict[str, float]) -> str:
    # Basic recommendations based on risk and certain conditions
    recs: List[str] = []

    if risk_label == 2:
        recs.append("Avoid travel; follow local emergency alerts.")
        if features.get("wind_speed_ms", 0) >= 25:
            recs.append("Secure loose outdoor items; stay away from windows.")
        if features.get("precipitation_mm", 0) >= 60:
            recs.append("Beware of flooding; move to higher ground if necessary.")
        if features.get("temperature_c", 0) <= -20:
            recs.append("Limit time outdoors; risk of frostbite. Dress in layers.")
        if features.get("temperature_c", 0) >= 40 and features.get("humidity_pct", 100) <= 15:
            recs.append("Extreme heat; stay hydrated and avoid direct sun exposure.")
    elif risk_label == 1:
        recs.append("Exercise caution; conditions may worsen.")
        if features.get("wind_speed_ms", 0) >= 15:
            recs.append("Monitor wind advisories; secure light objects.")
        if features.get("precipitation_mm", 0) >= 30:
            recs.append("Carry rain gear; check drainage around your home.")
        if features.get("temperature_c", 0) <= -10:
            recs.append("Dress warmly; icy surfaces possible.")
        if features.get("temperature_c", 0) >= 35:
            recs.append("Stay cool; check on vulnerable individuals.")
    else:
        recs.append("Normal conditions; stay informed and prepared.")

    return "\n".join(recs)


def build_parser(meta: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weather Anomaly Detection - CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    features: List[str] = meta.get("features", [])
    ranges: dict = meta.get("ranges", {})

    units = {
        "temperature_c": "°C",
        "humidity_pct": "%",
        "wind_speed_ms": "m/s",
        "precipitation_mm": "mm",
        "pressure_hpa": "hPa",
        "cloud_cover_pct": "%",
    }

    # Dynamically add feature arguments
    for key in features:
        rng = ranges.get(key)
        unit = units.get(key, "")
        help_parts = [f"Feature '{key}'"]
        if unit:
            help_parts.append(f"unit: {unit}")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            help_parts.append(f"range: {rng[0]}..{rng[1]}")
        parser.add_argument(f"--{key}", type=float, help=", ".join(help_parts))

    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    return parser


def prompt_missing(values: Dict[str, float], meta: dict) -> Dict[str, float]:
    features: List[str] = meta.get("features", [])
    ranges: dict = meta.get("ranges", {})

    def default_for(key: str) -> float:
        rng = ranges.get(key)
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                return float((float(rng[0]) + float(rng[1])) / 2)
            except Exception:
                pass
        return 0.0

    for key in features:
        if values.get(key) is None:
            dflt = default_for(key)
            while True:
                try:
                    raw = input(f"Enter {key} [{dflt}]: ").strip()
                    if raw == "":
                        values[key] = dflt
                        break
                    values[key] = float(raw)
                    break
                except ValueError:
                    print("Invalid number. Please try again.")
    return values


def main() -> int:
    try:
        model, meta = load_artifacts()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    parser = build_parser(meta)
    args = parser.parse_args()

    # Collect feature values
    features_order: List[str] = meta.get("features", [])
    provided = {k: getattr(args, k, None) for k in features_order}
    values = prompt_missing(provided, meta)

    # Prepare input DataFrame in training order (keeps feature names)
    X_df = pd.DataFrame([{k: float(values[k]) for k in features_order}], columns=features_order)

    # Predict
    proba = model.predict_proba(X_df)[0]
    pred = int(np.argmax(proba))

    classes = meta.get("classes", {"0": "Normal", "1": "Elevated", "2": "High"})
    risk_text = classes.get(str(pred), str(pred))

    # Build outputs
    prob_pairs = [(classes.get(str(i), str(i)), float(p)) for i, p in enumerate(proba)]
    recs_text = recommend_safety(pred, values)

    if getattr(args, "json", False):
        out = {
            "risk_label": pred,
            "risk_text": risk_text,
            "probabilities": {name: round(p, 6) for name, p in prob_pairs},
            "features": {k: float(values[k]) for k in features_order},
            "recommendations": recs_text.split("\n") if recs_text else [],
        }
        print(json.dumps(out, ensure_ascii=False))
    else:
        print("Predicted Risk:", risk_text)
        probs_str = ", ".join([f"{name}: {p:.2f}" for name, p in prob_pairs])
        print("Probabilities:", probs_str)
        print()
        print("Recommendations:")
        print(recs_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())