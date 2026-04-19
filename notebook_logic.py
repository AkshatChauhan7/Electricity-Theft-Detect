from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


NORMAL_LOSS_PCT = 10.0


@dataclass
class FeederMetrics:
    feeder_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    transformer_load_sum: float
    meter_load_sum: float
    energy_loss_kwh: float
    loss_pct: float
    avg_loss_pct: float
    max_loss_pct: float
    recent_loss_values: list[float]


@dataclass
class AnomalyResult:
    points: int
    duration_hours: float
    max_consecutive_points: int
    trend: str
    status: str
    risk_level: str
    deviation_from_normal: float


def _consecutive_count(series: pd.Series) -> pd.Series:
    count = 0
    result = []
    for value in series.tolist():
        if int(value) == 1:
            count += 1
        else:
            count = 0
        result.append(count)
    return pd.Series(result, index=series.index)


def _infer_granularity_hours(filtered_df: pd.DataFrame) -> float:
    if len(filtered_df) < 2:
        return 1.0
    diffs = filtered_df["timestamp"].diff().dropna().dt.total_seconds() / 3600.0
    if diffs.empty:
        return 1.0
    step = float(diffs.median())
    return step if step > 0 else 1.0


def compute_feeder_metrics(df: pd.DataFrame, feeder: str, start, end) -> Tuple[pd.DataFrame, FeederMetrics]:
    feeder_df = df[df["feeder_id"] == feeder].copy()
    feeder_df = feeder_df.sort_values("timestamp").reset_index(drop=True)

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    filtered_df = feeder_df[(feeder_df["timestamp"] >= start_ts) & (feeder_df["timestamp"] <= end_ts)].copy()
    if filtered_df.empty:
        raise ValueError("No data found for selected feeder and time window")

    filtered_df["loss_kwh"] = filtered_df["transformer_load_kWh"] - filtered_df["total_meter_load_kWh"]
    filtered_df["loss_pct"] = np.where(
        filtered_df["transformer_load_kWh"] > 0,
        (filtered_df["loss_kwh"] / filtered_df["transformer_load_kWh"]) * 100.0,
        0.0,
    )

    filtered_df["rolling_avg_loss"] = filtered_df["loss_pct"].rolling(window=3, min_periods=1).mean()
    filtered_df["rolling_std_loss"] = filtered_df["loss_pct"].rolling(window=3, min_periods=1).std().fillna(0.0)
    filtered_df["rolling_max_loss"] = filtered_df["loss_pct"].rolling(window=3, min_periods=1).max()
    filtered_df["high_loss_flag"] = (filtered_df["loss_pct"] > 15.0).astype(int)
    filtered_df["consecutive_high_loss_count"] = _consecutive_count(filtered_df["high_loss_flag"])
    filtered_df["loss_diff"] = filtered_df["loss_pct"].diff().fillna(0.0)

    transformer_sum = float(filtered_df["transformer_load_kWh"].sum())
    meter_sum = float(filtered_df["total_meter_load_kWh"].sum())
    loss_kwh_sum = max(transformer_sum - meter_sum, 0.0)
    window_loss_pct = (loss_kwh_sum / transformer_sum * 100.0) if transformer_sum > 0 else 0.0

    metrics = FeederMetrics(
        feeder_id=feeder,
        start_time=pd.to_datetime(filtered_df["timestamp"].min()),
        end_time=pd.to_datetime(filtered_df["timestamp"].max()),
        transformer_load_sum=transformer_sum,
        meter_load_sum=meter_sum,
        energy_loss_kwh=loss_kwh_sum,
        loss_pct=float(window_loss_pct),
        avg_loss_pct=float(filtered_df["loss_pct"].mean()),
        max_loss_pct=float(filtered_df["loss_pct"].max()),
        recent_loss_values=[round(v, 2) for v in filtered_df["loss_pct"].tail(5).tolist()],
    )

    return filtered_df, metrics


def detect_anomaly(filtered_df: pd.DataFrame, threshold_pct: float = 15.0) -> Tuple[pd.DataFrame, AnomalyResult]:
    detection_df = filtered_df.copy()
    detection_df["is_anomaly"] = (detection_df["loss_pct"] > threshold_pct).astype(int)

    points = int(detection_df["is_anomaly"].sum())
    max_consecutive = int(detection_df["is_anomaly"].groupby((detection_df["is_anomaly"] == 0).cumsum()).cumsum().max()) if points > 0 else 0

    step_hours = _infer_granularity_hours(detection_df)
    duration_hours = float(points * step_hours)

    trend = "stable"
    if points > 0:
        trend = "sustained" if max_consecutive >= 3 else "spike"

    latest_loss = float(detection_df["loss_pct"].iloc[-1])
    deviation_from_normal = latest_loss - NORMAL_LOSS_PCT

    status = "NORMAL"
    if latest_loss > threshold_pct:
        status = "THEFT" if max_consecutive >= 3 else "FAULT"

    risk_level = "LOW"
    if latest_loss >= 22:
        risk_level = "HIGH"
    elif latest_loss >= 12:
        risk_level = "MEDIUM"

    result = AnomalyResult(
        points=points,
        duration_hours=duration_hours,
        max_consecutive_points=max_consecutive,
        trend=trend,
        status=status,
        risk_level=risk_level,
        deviation_from_normal=round(deviation_from_normal, 2),
    )
    return detection_df, result


def build_openai_context(metrics: FeederMetrics, anomaly: AnomalyResult) -> Dict:
    return {
        "feeder_id": metrics.feeder_id,
        "loss_pct": round(metrics.loss_pct, 2),
        "deviation_from_normal": anomaly.deviation_from_normal,
        "duration_hours": round(anomaly.duration_hours, 2),
        "trend": anomaly.trend,
        "recent_loss_values": metrics.recent_loss_values,
        "anomaly_points": anomaly.points,
        "status": anomaly.status,
        "risk_level": anomaly.risk_level,
    }
