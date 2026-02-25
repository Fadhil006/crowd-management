"""
Streamlit Dashboard
===================
Real-time crowd management dashboard with:
  - Live density & velocity charts per zone
  - Congestion risk probability gauges
  - Color-coded risk indicators (Green/Yellow/Red)
  - Auto-updating digital signage messages
  - Scenario switching (Normal / Post-Event Rush / Emergency)

Updates every 2 seconds to simulate live data feed.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.simulate_data import (
    generate_normal_day,
    generate_post_event_rush,
    generate_emergency_evacuation,
)
from src.features import get_realtime_features, get_feature_columns, ROLLING_WINDOW
from src.predictor import (
    predict_zone,
    PredictionResult,
    YELLOW_THRESHOLD,
    RED_THRESHOLD,
)
from src.model import load_model


# ── Page Config ──
st.set_page_config(
    page_title="CrowdAI — Congestion Prediction",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .risk-card {
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 0.5rem;
        border: 2px solid;
    }
    .risk-green { background: #0d2818; border-color: #44BB44; }
    .risk-yellow { background: #2d2200; border-color: #FFAA00; }
    .risk-red { background: #2d0a0a; border-color: #FF4444; animation: pulse 1.5s infinite; }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .signage-box {
        padding: 1rem;
        border-radius: 8px;
        background: #1a1a2e;
        border-left: 4px solid #00d4ff;
        margin: 0.5rem 0;
        font-size: 1.1em;
    }
    .signage-active {
        border-left-color: #FF4444;
        background: #2d0a0a;
    }
    .metric-big {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .header-emoji { font-size: 1.5em; }
    div[data-testid="stMetric"] {
        background-color: #0e1117;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Scenario mapping ──
SCENARIOS = {
    "🏢 Normal Day": generate_normal_day,
    "🎉 Post-Event Rush": generate_post_event_rush,
    "🚨 Emergency Evacuation": generate_emergency_evacuation,
}


@st.cache_resource
def get_model():
    """Load the trained model (cached so it only loads once)."""
    return load_model()


def create_zone_chart(zone_history: pd.DataFrame, zone_id: str) -> go.Figure:
    """Create a dual-axis chart showing density and velocity over time."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=zone_history["timestamp"],
            y=zone_history["density"],
            name="Density",
            line=dict(color="#FF6B6B", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.1)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=zone_history["timestamp"],
            y=zone_history["velocity"],
            name="Velocity",
            line=dict(color="#4ECDC4", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(78,205,196,0.1)",
        ),
        secondary_y=True,
    )

    # Add congestion threshold line
    fig.add_hline(
        y=4.0, line_dash="dash", line_color="rgba(255,68,68,0.5)",
        annotation_text="Congestion Threshold",
        annotation_position="top left",
        secondary_y=False,
    )

    fig.update_layout(
        title=dict(text=f"📍 {zone_id}", font=dict(size=16)),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
    )
    fig.update_yaxes(title_text="Density (people/m²)", secondary_y=False, range=[0, 10])
    fig.update_yaxes(title_text="Velocity (m/s)", secondary_y=True, range=[0, 2.2])

    return fig


def create_risk_gauge(prediction: PredictionResult) -> go.Figure:
    """Create a gauge chart showing congestion risk probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction.risk_probability * 100,
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": prediction.risk_color},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 40], "color": "rgba(68,187,68,0.15)"},
                {"range": [40, 70], "color": "rgba(255,170,0,0.15)"},
                {"range": [70, 100], "color": "rgba(255,68,68,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": prediction.risk_probability * 100,
            },
        },
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_risk_card(prediction: PredictionResult):
    """Render a color-coded risk card for a zone."""
    css_class = f"risk-{prediction.risk_level}"
    ttc = (
        f"{prediction.time_to_congestion} min"
        if prediction.time_to_congestion > 0
        else "—"
    )

    st.markdown(f"""
    <div class="risk-card {css_class}">
        <h3>{prediction.zone_id}</h3>
        <div class="metric-big" style="color: {prediction.risk_color}">
            {prediction.risk_probability:.0%}
        </div>
        <p>Risk: <strong>{prediction.risk_level.upper()}</strong></p>
        <p>⏱️ Time to congestion: <strong>{ttc}</strong></p>
    </div>
    """, unsafe_allow_html=True)


def render_signage(prediction: PredictionResult):
    """Render digital signage message."""
    active_class = "signage-active" if prediction.signage_active else ""
    st.markdown(f"""
    <div class="signage-box {active_class}">
        <small>📺 Digital Signage — {prediction.zone_id}</small><br>
        <strong>{prediction.signage_message}</strong>
    </div>
    """, unsafe_allow_html=True)


def main():
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🏟️ CrowdAI Control")
        st.markdown("---")

        scenario = st.selectbox(
            "🎬 Simulation Scenario",
            list(SCENARIOS.keys()),
            help="Switch between different crowd scenarios",
        )

        update_speed = st.slider(
            "⏩ Update Speed (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="How often the dashboard refreshes",
        )

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.markdown("""
        - **Algorithm:** Logistic Regression
        - **Features:** 7 engineered features
        - **Prediction:** 10-15 min ahead
        - **Update rate:** Real-time (simulated)
        """)

        st.markdown("---")
        st.markdown("### 🎯 Risk Thresholds")
        st.markdown(f"""
        - 🟢 **Green:** < {YELLOW_THRESHOLD:.0%}
        - 🟡 **Yellow:** {YELLOW_THRESHOLD:.0%} – {RED_THRESHOLD:.0%}
        - 🔴 **Red:** > {RED_THRESHOLD:.0%}
        """)

        st.markdown("---")
        if st.button("🔄 Reset Simulation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Header ──
    st.markdown("""
    # 🏟️ CrowdAI — Congestion Prediction Dashboard
    **Privacy-first AI crowd management** · Predicting congestion 10-15 minutes before it happens
    """)

    # ── Load model ──
    try:
        model, scaler = get_model()
    except Exception:
        st.error("⚠️ Model not found! Please run training first: `python -m src.model`")
        st.stop()

    # ── Generate scenario data ──
    scenario_fn = SCENARIOS[scenario]
    np.random.seed(42)
    full_data = scenario_fn()
    zones = sorted(full_data["zone_id"].unique())

    # Prepare per-zone data
    zone_data = {}
    for zone in zones:
        zone_data[zone] = full_data[full_data["zone_id"] == zone].reset_index(drop=True)

    n_points = len(zone_data[zones[0]])

    # ── Session state for simulation step ──
    if "step" not in st.session_state or st.session_state.get("scenario") != scenario:
        st.session_state.step = ROLLING_WINDOW  # Start after enough history for features
        st.session_state.scenario = scenario

    step = st.session_state.step

    # ── Progress bar ──
    progress = step / n_points
    st.progress(progress, text=f"Simulation: {step}/{n_points} data points ({progress:.0%})")

    # ── Make predictions for each zone ──
    predictions: dict[str, PredictionResult] = {}
    zone_histories: dict[str, pd.DataFrame] = {}

    for zone in zones:
        # Get history up to current step
        history = zone_data[zone].iloc[max(0, step - 50):step + 1].copy()
        zone_histories[zone] = history

        # Compute features from history
        features = get_realtime_features(history)
        if features:
            predictions[zone] = predict_zone(zone, features, model, scaler)

    # ── Risk Overview Row ──
    st.markdown("### 🚦 Risk Overview")
    risk_cols = st.columns(3)
    for i, zone in enumerate(zones):
        if zone in predictions:
            with risk_cols[i]:
                render_risk_card(predictions[zone])

    # ── Charts Row ──
    st.markdown("### 📈 Live Sensor Data")
    chart_cols = st.columns(3)
    for i, zone in enumerate(zones):
        with chart_cols[i]:
            chart_data = zone_data[zone].iloc[:step + 1]
            fig = create_zone_chart(chart_data, zone)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{zone}_{step}")

    # ── Risk Gauges ──
    st.markdown("### 🎯 Congestion Probability")
    gauge_cols = st.columns(3)
    for i, zone in enumerate(zones):
        if zone in predictions:
            with gauge_cols[i]:
                fig = create_risk_gauge(predictions[zone])
                st.plotly_chart(fig, use_container_width=True, key=f"gauge_{zone}_{step}")

    # ── Digital Signage ──
    st.markdown("### 📺 Digital Signage Output")
    signage_cols = st.columns(3)
    for i, zone in enumerate(zones):
        if zone in predictions:
            with signage_cols[i]:
                render_signage(predictions[zone])

    # ── Auto-advance simulation ──
    if step < n_points - 1:
        time.sleep(update_speed)
        st.session_state.step = step + 1
        st.rerun()
    else:
        st.success("✅ Simulation complete! Switch scenario or reset to restart.")
        st.balloons()


if __name__ == "__main__":
    main()
