"""
Prediction Logic Module
=======================
Takes a trained model and live feature data → outputs actionable predictions.

Outputs:
  1. Risk Probability (0–1)
  2. Risk Level (Green / Yellow / Red)
  3. Estimated time-to-congestion (minutes)
  4. Signage message (if high risk)

Signage trigger logic:
  - prob > 0.7 → Trigger zone-specific redirect message
  - prob > 0.85 → Trigger emergency-level message
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    from src.model import load_model
    from src.features import get_feature_columns, get_realtime_features, PREDICTION_HORIZON
except ImportError:
    from model import load_model
    from features import get_feature_columns, get_realtime_features, PREDICTION_HORIZON


# ── Signage messages per zone ──
SIGNAGE_MESSAGES = {
    "Zone_A": {
        "warning": "⚠️ Zone A: Crowd building. Please consider Gate B.",
        "critical": "🚨 Zone A RESTRICTED — Redirect to Gate B immediately!",
        "emergency": "🆘 EMERGENCY: Zone A blocked. Use Alternate Exit NOW!",
    },
    "Zone_B": {
        "warning": "⚠️ Zone B: Moderate congestion ahead. Use side corridor.",
        "critical": "🚨 Zone B congested — Redirect to Zone C entrance!",
        "emergency": "🆘 EMERGENCY: Zone B blocked. Follow evacuation signs!",
    },
    "Zone_C": {
        "warning": "⚠️ Zone C: Increasing crowd density. Allow extra time.",
        "critical": "🚨 Zone C approaching capacity — Use Alternate Exit!",
        "emergency": "🆘 EMERGENCY: All zones critical. Follow staff directions!",
    },
}

# Risk thresholds
YELLOW_THRESHOLD = 0.4
RED_THRESHOLD = 0.7
EMERGENCY_THRESHOLD = 0.85


@dataclass
class PredictionResult:
    """Structured prediction output for one zone."""
    zone_id: str
    risk_probability: float
    risk_level: str         # "green", "yellow", "red"
    risk_color: str         # hex color for UI
    time_to_congestion: float  # minutes, -1 if no risk
    signage_message: str    # empty if no action needed
    signage_active: bool


def get_risk_level(prob: float) -> tuple[str, str]:
    """Map probability to risk level and color."""
    if prob >= RED_THRESHOLD:
        return "red", "#FF4444"
    elif prob >= YELLOW_THRESHOLD:
        return "yellow", "#FFAA00"
    else:
        return "green", "#44BB44"


def estimate_time_to_congestion(
    prob: float,
    density_rate: float,
    current_density: float,
    interval_seconds: int = 30,
) -> float:
    """
    Estimate minutes until congestion based on:
    - Current risk probability
    - Rate of density change
    - Distance from density threshold

    Returns -1 if no congestion is expected.
    """
    if prob < YELLOW_THRESHOLD:
        return -1.0

    density_threshold = 4.0  # Same as in features.py

    if density_rate <= 0.01:
        # Density not increasing — can't estimate time
        if prob >= RED_THRESHOLD:
            return 2.0  # Already near/at congestion
        return -1.0

    # Steps until density reaches threshold
    density_gap = max(density_threshold - current_density, 0.1)
    steps_to_threshold = density_gap / max(density_rate, 0.01)
    minutes_to_congestion = (steps_to_threshold * interval_seconds) / 60.0

    # Clamp to reasonable range
    minutes_to_congestion = max(0.5, min(minutes_to_congestion, 30.0))

    # If already high risk, override with short time
    if prob >= EMERGENCY_THRESHOLD:
        minutes_to_congestion = min(minutes_to_congestion, 2.0)
    elif prob >= RED_THRESHOLD:
        minutes_to_congestion = min(minutes_to_congestion, 8.0)

    return round(minutes_to_congestion, 1)


def get_signage_message(zone_id: str, prob: float) -> tuple[str, bool]:
    """
    Get the appropriate digital signage message for a zone based on risk.
    Returns (message, is_active).
    """
    zone_messages = SIGNAGE_MESSAGES.get(zone_id, SIGNAGE_MESSAGES["Zone_A"])

    if prob >= EMERGENCY_THRESHOLD:
        return zone_messages["emergency"], True
    elif prob >= RED_THRESHOLD:
        return zone_messages["critical"], True
    elif prob >= YELLOW_THRESHOLD:
        return zone_messages["warning"], True
    else:
        return "✅ Normal flow. No action required.", False


def predict_zone(
    zone_id: str,
    features: dict,
    model=None,
    scaler=None,
) -> PredictionResult:
    """
    Make a full prediction for one zone given its current features.
    Returns a structured PredictionResult.
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    feature_cols = get_feature_columns()
    feature_vector = np.array([[features.get(col, 0) for col in feature_cols]])

    # Scale features and predict probability
    feature_scaled = scaler.transform(feature_vector)
    prob = model.predict_proba(feature_scaled)[0][1]

    # Derive all outputs
    risk_level, risk_color = get_risk_level(prob)
    time_to_cong = estimate_time_to_congestion(
        prob=prob,
        density_rate=features.get("density_rate_of_change", 0),
        current_density=features.get("density", 0),
    )
    message, signage_active = get_signage_message(zone_id, prob)

    return PredictionResult(
        zone_id=zone_id,
        risk_probability=round(prob, 4),
        risk_level=risk_level,
        risk_color=risk_color,
        time_to_congestion=time_to_cong,
        signage_message=message,
        signage_active=signage_active,
    )


def predict_all_zones(
    zone_histories: dict[str, pd.DataFrame],
    model=None,
    scaler=None,
) -> dict[str, PredictionResult]:
    """
    Predict congestion risk for all zones given their recent history.
    zone_histories: {zone_id: DataFrame of recent readings}
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    results = {}
    for zone_id, history_df in zone_histories.items():
        features = get_realtime_features(history_df)
        if features:
            results[zone_id] = predict_zone(zone_id, features, model, scaler)

    return results


if __name__ == "__main__":
    # Quick test with synthetic features
    model, scaler = load_model()

    # Simulate a normal situation
    normal_features = {
        "density": 1.2,
        "velocity": 1.5,
        "rolling_density_mean": 1.1,
        "rolling_velocity_mean": 1.5,
        "density_rate_of_change": 0.02,
        "velocity_rate_of_change": -0.01,
        "density_velocity_ratio": 0.8,
    }

    # Simulate a pre-congestion situation
    risky_features = {
        "density": 3.8,
        "velocity": 0.6,
        "rolling_density_mean": 3.2,
        "rolling_velocity_mean": 0.8,
        "density_rate_of_change": 0.35,
        "velocity_rate_of_change": -0.15,
        "density_velocity_ratio": 6.3,
    }

    print("── Normal Situation ──")
    result = predict_zone("Zone_A", normal_features, model, scaler)
    print(f"   Risk: {result.risk_probability:.1%} ({result.risk_level})")
    print(f"   Signage: {result.signage_message}")

    print("\n── Pre-Congestion Situation ──")
    result = predict_zone("Zone_A", risky_features, model, scaler)
    print(f"   Risk: {result.risk_probability:.1%} ({result.risk_level})")
    print(f"   Time to congestion: {result.time_to_congestion} min")
    print(f"   Signage: {result.signage_message}")
