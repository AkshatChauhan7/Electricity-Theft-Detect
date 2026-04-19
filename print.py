import pandas as pd
import streamlit as st

from genai_utils import action_panel_ai_note, ask_grid, explain_anomaly, recommend_action, short_summary
from logic_utils import (
	DEFAULT_RISK_HIGH_THRESHOLD_PCT,
	DEFAULT_RISK_MEDIUM_THRESHOLD_PCT,
	DEFAULT_TARIFF_PER_UNIT,
	DEFAULT_THEFT_THRESHOLD_PCT,
	LOSS_ALERT_THRESHOLD_PCT,
	build_anomaly_data,
	build_anomaly_payload,
	classify_status,
	compute_impact_metrics,
	compute_window_metrics,
	get_anomaly_stats,
	infer_possible_cause,
	suggested_inspection_time,
)
from ui_components import inject_styles, plot_load_chart, plot_loss_chart, status_chip


@st.cache_data
def load_data() -> pd.DataFrame:
	data = pd.read_csv("enhanced_feeder_6month.csv")
	data["timestamp"] = pd.to_datetime(data["timestamp"])
	return data.sort_values("timestamp").reset_index(drop=True)


def get_ai_bundle(anomaly_data: dict) -> dict:
	cache_key = (
		anomaly_data["feeder_id"],
		anomaly_data["loss_pct"],
		anomaly_data["duration_hours"],
		anomaly_data["trend"],
	)
	if "ai_cache" not in st.session_state:
		st.session_state["ai_cache"] = {}
	if cache_key not in st.session_state["ai_cache"]:
		st.session_state["ai_cache"][cache_key] = {
			"explanation": explain_anomaly(anomaly_data),
			"action": recommend_action(anomaly_data),
			"summary": short_summary(anomaly_data),
		}
	return st.session_state["ai_cache"][cache_key]


st.set_page_config(page_title="Grid Intelligence Platform", layout="wide")
inject_styles()

st.title("⚡ Grid Intelligence Platform")
st.caption("AI-powered anomaly detection, explanation, and operational decision support.")

df = load_data()

st.sidebar.header("⚙️ Dynamic Controls")
fault_threshold_pct = st.sidebar.slider("Fault threshold (%)", min_value=8.0, max_value=25.0, value=float(LOSS_ALERT_THRESHOLD_PCT), step=0.5)
theft_threshold_pct = st.sidebar.slider("Theft threshold (%)", min_value=10.0, max_value=40.0, value=float(DEFAULT_THEFT_THRESHOLD_PCT), step=0.5)
risk_medium_threshold_pct = st.sidebar.slider("Risk MEDIUM threshold (%)", min_value=5.0, max_value=25.0, value=float(DEFAULT_RISK_MEDIUM_THRESHOLD_PCT), step=0.5)
risk_high_threshold_pct = st.sidebar.slider("Risk HIGH threshold (%)", min_value=10.0, max_value=40.0, value=float(DEFAULT_RISK_HIGH_THRESHOLD_PCT), step=0.5)
sustained_duration_threshold = st.sidebar.slider("Sustained anomaly threshold (hours)", min_value=1, max_value=8, value=3, step=1)
tariff_per_unit = st.sidebar.number_input("Tariff (₹ per unit)", min_value=1.0, max_value=25.0, value=float(DEFAULT_TARIFF_PER_UNIT), step=0.5)
chat_context_rows = st.sidebar.slider("Chat context rows", min_value=6, max_value=96, value=24, step=6)

left, right = st.columns([1.2, 2])
with left:
	feeders = sorted(df["feeder_id"].unique())
	feeder = st.selectbox("🔌 Feeder", feeders)
with right:
	max_rows = int(df[df["feeder_id"] == feeder].shape[0])
	default_rows = min(168, max_rows)
	window_size = st.slider("🕒 Time window (latest records)", min_value=24, max_value=max(24, max_rows), value=max(24, default_rows), step=24)

feeder_df = df[df["feeder_id"] == feeder].sort_values("timestamp").reset_index(drop=True)
if feeder_df.empty:
	st.error("No data found for selected feeder.")
	st.stop()

window_df = feeder_df.tail(window_size).copy().reset_index(drop=True)

window_df["loss_pct"] = (
	((window_df["transformer_load_kWh"] - window_df["total_meter_load_kWh"]) / window_df["transformer_load_kWh"].replace(0, pd.NA))
	.fillna(0.0)
)
window_df["loss_pct_display"] = window_df["loss_pct"] * 100

window_metrics = compute_window_metrics(window_df)
anomaly_stats = get_anomaly_stats(window_df, anomaly_threshold_pct=fault_threshold_pct)

window_df["status_pred"] = window_df.apply(
	lambda row: classify_status(
		float(row["loss_pct"]) * 100,
		int(row["high_loss_flag"]),
		int(row["consecutive_high_loss_count"]),
		fault_threshold_pct=fault_threshold_pct,
		theft_threshold_pct=theft_threshold_pct,
		sustained_duration_threshold=sustained_duration_threshold,
	),
	axis=1,
)
window_df["is_anomaly"] = window_df["loss_pct_display"] > fault_threshold_pct

latest_row = window_df.iloc[-1]
loss_pct = float(window_metrics.loss_pct)
status_pred = latest_row["status_pred"]

impact = compute_impact_metrics(
	transformer_load=window_metrics.transformer_load_kwh,
	meter_load=window_metrics.meter_load_kwh,
	loss_pct=loss_pct,
	tariff_per_unit=tariff_per_unit,
	risk_medium_threshold_pct=risk_medium_threshold_pct,
	risk_high_threshold_pct=risk_high_threshold_pct,
)

st.markdown("### 1) Problem & Detection")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Window Loss %", f"{loss_pct:.2f}%")
col2.metric("Predicted Status", status_pred)
col3.metric("Risk Level", impact.risk_level)
col4.metric("Estimated Revenue Loss", f"₹ {impact.revenue_loss_inr:,.2f}")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Transformer Load (sum)", f"{window_metrics.transformer_load_kwh:,.2f} kWh")
d2.metric("Meter Load (sum)", f"{window_metrics.meter_load_kwh:,.2f} kWh")
d3.metric("Anomaly Points", anomaly_stats.anomaly_points)
d4.metric("Anomaly Duration", f"{anomaly_stats.anomaly_duration_hours:.1f} h")

alert_label = "🟢 Stable" if status_pred == "NORMAL" else ("🟡 Needs Attention" if status_pred == "FAULT" else "🔴 Theft Risk")
st.markdown(
	f"{alert_label} &nbsp;&nbsp; Status: {status_chip(status_pred)} &nbsp;&nbsp; Risk: {status_chip(impact.risk_level)}",
	unsafe_allow_html=True,
)

st.caption(f"Trend detected from selected window: {anomaly_stats.trend.upper()} (max consecutive anomaly hours: {anomaly_stats.max_consecutive_hours})")

chart_left, chart_right = st.columns(2)
with chart_left:
	st.altair_chart(plot_load_chart(window_df), use_container_width=True)
with chart_right:
	st.altair_chart(plot_loss_chart(window_df, alert_threshold_pct=fault_threshold_pct), use_container_width=True)

st.markdown("### 2) AI Insights")
is_anomaly = status_pred != "NORMAL" or loss_pct > fault_threshold_pct

if is_anomaly:
	anomaly_data = build_anomaly_payload(feeder, window_metrics, anomaly_stats)
	ai_bundle = get_ai_bundle(anomaly_data)

	i1, i2, i3 = st.columns([1.2, 1.2, 1])
	with i1:
		st.subheader("🧠 Explanation")
		st.write(ai_bundle["explanation"])
	with i2:
		st.subheader("📌 Recommendation")
		st.write(ai_bundle["action"])
	with i3:
		st.subheader("📊 1-Line Summary")
		st.info(ai_bundle["summary"])
else:
	anomaly_data = build_anomaly_payload(feeder, window_metrics, anomaly_stats)
	st.success("No active anomaly in the selected window. System is operating within expected limits.")

st.markdown("### 3) Action Panel (Decision Support)")
inspection_time = suggested_inspection_time(impact.risk_level, is_peak_hour=int(latest_row["is_peak_hour"]))
possible_cause = infer_possible_cause(status_pred, latest_row)
ai_action_note = action_panel_ai_note(anomaly_data, impact.risk_level, possible_cause)

a1, a2, a3 = st.columns(3)
a1.metric("Suggested Inspection Time", inspection_time)
a2.metric("Priority", impact.risk_level)
a3.metric("Possible Cause", possible_cause)
st.write("🤖 Action Brief:")
st.write(ai_action_note)

st.markdown("### 4) Impact")
b1, b2, b3 = st.columns(3)
b1.metric("Estimated Energy Loss", f"{impact.loss_kwh:.2f} kWh")
b2.metric("Estimated Revenue Loss", f"₹ {impact.revenue_loss_inr:,.2f}")
b3.metric("Alert Threshold", f"{fault_threshold_pct:.1f}%")

st.markdown("### 5) Ask the Grid")
user_query = st.text_input("Ask about this feeder")
if user_query:
	context_df = window_df.tail(chat_context_rows).copy()
	context_df["loss_pct_display"] = context_df["loss_pct"] * 100
	context_records = context_df[
		["timestamp", "transformer_load_kWh", "total_meter_load_kWh", "loss_pct_display", "status_pred", "is_anomaly"]
	].copy()
	context_records["timestamp"] = context_records["timestamp"].astype(str)
	context_payload = context_records.round(3).to_dict("records")

	feeder_snapshot = {
		"window_start": str(window_df["timestamp"].min()),
		"window_end": str(window_df["timestamp"].max()),
		"window_loss_pct": round(loss_pct, 2),
		"risk_level": impact.risk_level,
		"anomaly_points": anomaly_stats.anomaly_points,
		"anomaly_duration_hours": anomaly_stats.anomaly_duration_hours,
		"trend": anomaly_stats.trend,
		"recent_rows": context_payload,
	}
	answer = ask_grid(user_query, anomaly_data, feeder_snapshot)
	st.write(answer)