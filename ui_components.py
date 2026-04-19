from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .main > div {padding-top: 1rem;}
            .section-card {
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 14px;
                padding: 1rem 1rem 0.7rem 1rem;
                margin-bottom: 0.9rem;
            }
            .kpi-card {
                background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
                border: 1px solid #374151;
                border-radius: 12px;
                padding: 0.8rem;
            }
            .chip {
                display: inline-block;
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_chip(label: str) -> str:
    color_map = {
        "NORMAL": ("#14532d", "#86efac"),
        "FAULT": ("#78350f", "#fde68a"),
        "THEFT": ("#7f1d1d", "#fca5a5"),
        "LOW": ("#14532d", "#86efac"),
        "MEDIUM": ("#78350f", "#fde68a"),
        "HIGH": ("#7f1d1d", "#fca5a5"),
    }
    bg, fg = color_map.get(label, ("#1f2937", "#d1d5db"))
    return f'<span class="chip" style="background:{bg}; color:{fg};">{label}</span>'


def plot_load_chart(window_df: pd.DataFrame) -> alt.Chart:
    data = window_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    long_df = data.melt(
        id_vars=["timestamp", "status_pred"],
        value_vars=["transformer_load_kWh", "total_meter_load_kWh"],
        var_name="series",
        value_name="kwh",
    )

    series_color = alt.Scale(
        domain=["transformer_load_kWh", "total_meter_load_kWh"],
        range=["#60a5fa", "#34d399"],
    )

    line = (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("kwh:Q", title="Load (kWh)"),
            color=alt.Color("series:N", scale=series_color, title="Series"),
        )
    )

    anomaly_points = (
        alt.Chart(data[data["is_anomaly"]])
        .mark_circle(size=55, color="#ef4444")
        .encode(x="timestamp:T", y=alt.Y("transformer_load_kWh:Q", title="Load (kWh)"))
        .properties(title="Transformer vs Meter Load (anomaly markers in red)")
    )

    return (line + anomaly_points).interactive()


def plot_loss_chart(window_df: pd.DataFrame, alert_threshold_pct: float) -> alt.Chart:
    data = window_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["loss_pct_display"] = data["loss_pct"] * 100

    loss_line = (
        alt.Chart(data)
        .mark_line(strokeWidth=2, color="#f59e0b")
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("loss_pct_display:Q", title="Loss %"),
            tooltip=["timestamp:T", "loss_pct_display:Q", "status_pred:N", "is_anomaly:N"],
        )
    )

    threshold = pd.DataFrame({"y": [alert_threshold_pct]})
    threshold_line = (
        alt.Chart(threshold)
        .mark_rule(color="#94a3b8", strokeDash=[5, 5])
        .encode(y="y:Q")
    )

    anomaly_points = (
        alt.Chart(data[data["is_anomaly"]])
        .mark_circle(size=70, color="#ef4444")
        .encode(x="timestamp:T", y="loss_pct_display:Q")
    )

    return (loss_line + threshold_line + anomaly_points).properties(
        title="Loss % Trend with Alert Threshold"
    ).interactive()
