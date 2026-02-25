"""
Feature Engineering Module
==========================
Transforms raw sensor data (density, velocity) into ML-ready features.

Why each feature matters:
─────────────────────────
1. rolling_density_mean (5-point window = ~2.5 min)
   → Smooths sensor noise. A rising rolling average signals sustained crowd buildup,
     not just a momentary blip.

2. rolling_velocity_mean (5-point window)
   → Smooths velocity readings. A falling rolling average means people are
     consistently slowing down — a hallmark of developing congestion.

3. density_rate_of_change (1st derivative)
   → How fast density is increasing RIGHT NOW. A high positive value means
     the crowd is building rapidly — urgent signal for prediction.

4. velocity_rate_of_change (1st derivative)
   → How fast velocity is decreasing. A large negative value means people
     are rapidly slowing — strong congestion precursor.

5. density_velocity_ratio
   → Combined signal: high density + low velocity = congestion.
     This ratio captures the interaction between both variables.

6. congestion_label (binary target)
   → Ground truth for training. Defined as:
     density > 4.0 AND velocity < 0.5
     These thresholds come from crowd dynamics research.

7. pre_congestion_label (what we actually predict)
   → Shifted forward by PREDICTION_HORIZON steps.
     "Will congestion happen in the next 10-15 minutes?"
     This is what makes it a PREDICTION system, not just detection.
"""

import numpy as np
import pandas as pd

# Prediction horizon: how many steps ahead to predict
# At 30-sec intervals, 20 steps = 10 minutes, 30 steps = 15 minutes
PREDICTION_HORIZON = 25  # ~12.5 minutes ahead
ROLLING_WINDOW = 5       # 5 × 30s = 2.5 minute smoothing window

# Congestion thresholds (from crowd dynamics research)
DENSITY_THRESHOLD = 4.0   # people/m² — above this is "crowded"
VELOCITY_THRESHOLD = 0.5  # m/s — below this people are nearly stuck


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw sensor data and add engineered features for each zone independently.
    Returns a new DataFrame with all features added.
    """
    result_frames = []

    for zone_id in df["zone_id"].unique():
        zone_df = df[df["zone_id"] == zone_id].copy()
        zone_df = zone_df.sort_values("timestamp").reset_index(drop=True)

        # ── Feature 1 & 2: Rolling averages ──
        zone_df["rolling_density_mean"] = (
            zone_df["density"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        zone_df["rolling_velocity_mean"] = (
            zone_df["velocity"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )

        # ── Feature 3 & 4: Rate of change (1st derivative) ──
        zone_df["density_rate_of_change"] = zone_df["density"].diff().fillna(0)
        zone_df["velocity_rate_of_change"] = zone_df["velocity"].diff().fillna(0)

        # ── Feature 5: Density-to-velocity ratio ──
        zone_df["density_velocity_ratio"] = (
            zone_df["density"] / zone_df["velocity"].clip(lower=0.05)
        )

        # ── Target: Current congestion label ──
        zone_df["congestion_label"] = (
            (zone_df["density"] > DENSITY_THRESHOLD) &
            (zone_df["velocity"] < VELOCITY_THRESHOLD)
        ).astype(int)

        # ── Prediction target: Will congestion happen in next N steps? ──
        # Shift the congestion label BACKWARD so current row predicts future
        zone_df["pre_congestion_label"] = (
            zone_df["congestion_label"]
            .rolling(window=PREDICTION_HORIZON, min_periods=1)
            .max()
            .shift(-PREDICTION_HORIZON)
            .fillna(0)
            .astype(int)
        )

        result_frames.append(zone_df)

    result = pd.concat(result_frames, ignore_index=True)
    return result


def get_feature_columns() -> list[str]:
    """Return the list of feature column names used for model input."""
    return [
        "density",
        "velocity",
        "rolling_density_mean",
        "rolling_velocity_mean",
        "density_rate_of_change",
        "velocity_rate_of_change",
        "density_velocity_ratio",
    ]


def get_realtime_features(history_df: pd.DataFrame) -> dict:
    """
    Given a short history of recent readings for ONE zone,
    compute the current feature vector for real-time prediction.
    Used by the Streamlit dashboard during live simulation.
    """
    if len(history_df) == 0:
        return {}

    latest = history_df.iloc[-1]
    window = history_df.tail(ROLLING_WINDOW)

    features = {
        "density": latest["density"],
        "velocity": latest["velocity"],
        "rolling_density_mean": window["density"].mean(),
        "rolling_velocity_mean": window["velocity"].mean(),
        "density_rate_of_change": (
            history_df["density"].diff().iloc[-1] if len(history_df) > 1 else 0
        ),
        "velocity_rate_of_change": (
            history_df["velocity"].diff().iloc[-1] if len(history_df) > 1 else 0
        ),
        "density_velocity_ratio": (
            latest["density"] / max(latest["velocity"], 0.05)
        ),
    }
    return features


if __name__ == "__main__":
    from simulate_data import generate_training_dataset

    df = generate_training_dataset()
    featured = engineer_features(df)

    print(f"Dataset shape: {featured.shape}")
    print(f"\nFeature columns: {get_feature_columns()}")
    print(f"\nCongestion events (current): {featured['congestion_label'].sum()}")
    print(f"Pre-congestion labels (prediction target): {featured['pre_congestion_label'].sum()}")
    print(f"\nClass balance:\n{featured['pre_congestion_label'].value_counts(normalize=True)}")
    print(f"\nSample with features:\n{featured.head(10)}")
