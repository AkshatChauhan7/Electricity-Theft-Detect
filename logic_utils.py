from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

LOSS_ALERT_THRESHOLD_PCT = 15.0
DEFAULT_THEFT_THRESHOLD_PCT = 22.0
DEFAULT_RISK_MEDIUM_THRESHOLD_PCT = 12.0
DEFAULT_RISK_HIGH_THRESHOLD_PCT = 22.0
DEFAULT_TARIFF_PER_UNIT = 8.0


@dataclass
class ImpactMetrics:
    loss_kwh: float
    revenue_loss_inr: float
    risk_level: str


@dataclass
class WindowMetrics:
    transformer_load_kwh: float
    meter_load_kwh: float
    loss_kwh: float
    loss_pct: float


@dataclass
class AnomalyStats:
    anomaly_points: int
    anomaly_duration_hours: float
    max_consecutive_hours: int
    trend: str


def classify_status(
    loss_pct: float,
    high_loss_flag: int,
    consecutive_high_loss_count: int,
    fault_threshold_pct: float = LOSS_ALERT_THRESHOLD_PCT,
    theft_threshold_pct: float = DEFAULT_THEFT_THRESHOLD_PCT,
    sustained_duration_threshold: int = 2,
) -> str:
    """Rule-based status classification: NORMAL / THEFT / FAULT."""
    if loss_pct >= theft_threshold_pct or (high_loss_flag == 1 and consecutive_high_loss_count >= sustained_duration_threshold):
        return "THEFT"
    if loss_pct >= fault_threshold_pct or high_loss_flag == 1:
        return "FAULT"
    return "NORMAL"


def get_risk_level(
    loss_pct: float,
    medium_threshold_pct: float = DEFAULT_RISK_MEDIUM_THRESHOLD_PCT,
    high_threshold_pct: float = DEFAULT_RISK_HIGH_THRESHOLD_PCT,
) -> str:
    if loss_pct >= high_threshold_pct:
        return "HIGH"
    if loss_pct >= medium_threshold_pct:
        return "MEDIUM"
    return "LOW"


def build_anomaly_data(row, feeder_id: str) -> dict:
    loss_pct = float(row["loss_pct"]) * 100
    duration = float(row.get("consecutive_high_loss_count", 0))
    trend = "sustained" if duration >= 2 else "spiky"
    return {
        "feeder_id": feeder_id,
        "loss_pct": round(loss_pct, 2),
        "duration_hours": round(duration, 2),
        "trend": trend,
    }


def compute_window_metrics(window_df: pd.DataFrame) -> WindowMetrics:
    transformer_load = float(window_df["transformer_load_kWh"].sum())
    meter_load = float(window_df["total_meter_load_kWh"].sum())
    loss_kwh = max(transformer_load - meter_load, 0.0)
    loss_pct = (loss_kwh / transformer_load * 100.0) if transformer_load > 0 else 0.0
    return WindowMetrics(
        transformer_load_kwh=transformer_load,
        meter_load_kwh=meter_load,
        loss_kwh=loss_kwh,
        loss_pct=loss_pct,
    )


def get_anomaly_stats(window_df: pd.DataFrame, anomaly_threshold_pct: float) -> AnomalyStats:
    work_df = window_df.copy()
    work_df["loss_pct_display"] = work_df["loss_pct"] * 100
    work_df["is_anomaly"] = work_df["loss_pct_display"] > anomaly_threshold_pct

    anomaly_points = int(work_df["is_anomaly"].sum())
    anomaly_duration_hours = float(anomaly_points)

    max_consecutive = 0
    current = 0
    for value in work_df["is_anomaly"].tolist():
        if value:
            current += 1
            if current > max_consecutive:
                max_consecutive = current
        else:
            current = 0

    trend = "sustained" if max_consecutive >= 3 else "spike"
    if anomaly_points == 0:
        trend = "stable"

    return AnomalyStats(
        anomaly_points=anomaly_points,
        anomaly_duration_hours=anomaly_duration_hours,
        max_consecutive_hours=max_consecutive,
        trend=trend,
    )


def build_anomaly_payload(feeder_id: str, window_metrics: WindowMetrics, anomaly_stats: AnomalyStats) -> Dict:
    return {
        "feeder_id": feeder_id,
        "loss_pct": round(window_metrics.loss_pct, 2),
        "duration_hours": round(anomaly_stats.anomaly_duration_hours, 2),
        "trend": anomaly_stats.trend,
    }


def compute_impact_metrics(
    transformer_load: float,
    meter_load: float,
    loss_pct: float,
    tariff_per_unit: float = DEFAULT_TARIFF_PER_UNIT,
    risk_medium_threshold_pct: float = DEFAULT_RISK_MEDIUM_THRESHOLD_PCT,
    risk_high_threshold_pct: float = DEFAULT_RISK_HIGH_THRESHOLD_PCT,
) -> ImpactMetrics:
    loss_kwh = max(transformer_load - meter_load, 0.0)
    revenue_loss_inr = loss_kwh * tariff_per_unit
    risk_level = get_risk_level(
        loss_pct,
        medium_threshold_pct=risk_medium_threshold_pct,
        high_threshold_pct=risk_high_threshold_pct,
    )
    return ImpactMetrics(loss_kwh=loss_kwh, revenue_loss_inr=revenue_loss_inr, risk_level=risk_level)


def suggested_inspection_time(risk_level: str, is_peak_hour: int = 0) -> str:
    if risk_level == "HIGH":
        return "Within 1 hour" if is_peak_hour == 1 else "Within 2 hours"
    if risk_level == "MEDIUM":
        return "Within 4 hours" if is_peak_hour == 1 else "Today (next maintenance slot)"
    return "This week"


def infer_possible_cause(status_pred: str, row) -> str:
    if status_pred == "THEFT":
        if float(row.get("consecutive_high_loss_count", 0)) >= 2:
            return "Sustained unauthorized tapping or bypass"
        return "Potential meter tampering"
    if status_pred == "FAULT":
        if float(row.get("ambient_temp_C", 0)) >= 35:
            return "Overheating or weather-induced technical fault"
        return "Meter mismatch / transformer-side technical issue"
    return "No immediate issue"
