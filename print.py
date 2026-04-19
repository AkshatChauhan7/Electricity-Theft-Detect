import pandas as pd
import streamlit as st

from genai_utils import action_panel_ai_note, generate_insights
from notebook_logic import build_openai_context, compute_feeder_metrics, detect_anomaly
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
		anomaly_data["deviation_from_normal"],
		anomaly_data["duration_hours"],
		anomaly_data["trend"],
		tuple(anomaly_data["recent_loss_values"]),
	)
	if "ai_cache" not in st.session_state:
		st.session_state["ai_cache"] = {}
	if cache_key not in st.session_state["ai_cache"]:
		st.session_state["ai_cache"][cache_key] = generate_insights(anomaly_data)
	return st.session_state["ai_cache"][cache_key]


st.set_page_config(page_title="Grid Intelligence Platform", layout="wide")
inject_styles()

st.title("⚡ Grid Intelligence Platform")
st.caption("AI-powered anomaly detection, explanation, and operational decision support.")

df = load_data()

st.sidebar.header("⚙️ Dynamic Controls")
anomaly_threshold_pct = st.sidebar.slider("Anomaly threshold (%)", min_value=8.0, max_value=25.0, value=15.0, step=0.5)
tariff_per_unit = st.sidebar.number_input("Tariff (₹ per unit)", min_value=1.0, max_value=25.0, value=8.0, step=0.5)

left, right = st.columns([1.2, 2])
with left:
	feeders = sorted(df["feeder_id"].unique())
	feeder = st.selectbox("🔌 Feeder", feeders)
with right:
	feeder_times = df[df["feeder_id"] == feeder]["timestamp"].sort_values()
	min_t = feeder_times.min()
	max_t = feeder_times.max()
	time_range = st.slider("🕒 Time window", min_value=min_t.to_pydatetime(), max_value=max_t.to_pydatetime(), value=(min_t.to_pydatetime(), max_t.to_pydatetime()))

try:
	window_df, metrics = compute_feeder_metrics(df, feeder, time_range[0], time_range[1])
except ValueError as exc:
	st.error(str(exc))
	st.stop()

window_df, anomaly = detect_anomaly(window_df, threshold_pct=anomaly_threshold_pct)
window_df["status_pred"] = window_df["is_anomaly"].map({True: "ANOMALY", False: "NORMAL"})
window_df["is_anomaly"] = window_df["is_anomaly"].astype(bool)

loss_pct = float(metrics.loss_pct)
status_pred = anomaly.status
revenue_loss = metrics.energy_loss_kwh * float(tariff_per_unit)

st.markdown("### 1) Problem & Detection")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Window Loss %", f"{loss_pct:.2f}%")
col2.metric("Predicted Status", status_pred)
col3.metric("Risk Level", anomaly.risk_level)
col4.metric("Estimated Revenue Loss", f"₹ {revenue_loss:,.2f}")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Transformer Load (sum)", f"{metrics.transformer_load_sum:,.2f} kWh")
d2.metric("Meter Load (sum)", f"{metrics.meter_load_sum:,.2f} kWh")
d3.metric("Anomaly Points", anomaly.points)
d4.metric("Anomaly Duration", f"{anomaly.duration_hours:.1f} h")

alert_label = "🟢 Stable" if status_pred == "NORMAL" else ("🟡 Needs Attention" if status_pred == "FAULT" else "🔴 Theft Risk")
st.markdown(
	f"{alert_label} &nbsp;&nbsp; Status: {status_chip(status_pred)} &nbsp;&nbsp; Risk: {status_chip(anomaly.risk_level)}",
	unsafe_allow_html=True,
)

st.caption(f"Trend detected from selected window: {anomaly.trend.upper()} (max consecutive anomaly points: {anomaly.max_consecutive_points})")
st.caption(f"Selected range: {metrics.start_time} → {metrics.end_time}")

chart_left, chart_right = st.columns(2)
with chart_left:
	st.altair_chart(plot_load_chart(window_df), width="stretch")
with chart_right:
	st.altair_chart(plot_loss_chart(window_df, alert_threshold_pct=anomaly_threshold_pct), width="stretch")

st.markdown("### 2) AI Insights")
anomaly_data = build_openai_context(metrics, anomaly)
ai_bundle = get_ai_bundle(anomaly_data)
ai_mode = "OpenAI" if ai_bundle.get("ai_source") == "openai" else "Fallback (rule-based)"
st.caption(f"Insight mode: {ai_mode}")

i1, i2, i3 = st.columns([1.2, 1.2, 1])
with i1:
	st.subheader("🧠 Explanation")
	st.write(ai_bundle["explanation"])
	st.caption(f"Root cause: {ai_bundle['root_cause_analysis']}")
with i2:
	st.subheader("📌 Recommended Action")
	st.write(ai_bundle["recommended_action"])
with i3:
	st.subheader("🚨 Severity")
	st.metric("AI Severity", str(ai_bundle["severity"]).upper())
	st.subheader("📊 Summary")
	st.info(ai_bundle["summary"])

st.markdown("### 3) Action Panel (Decision Support)")
inspection_time = "Within 2 hours" if anomaly.risk_level == "HIGH" else ("Today (next maintenance slot)" if anomaly.risk_level == "MEDIUM" else "This week")
possible_cause = "Likely theft pattern" if anomaly.trend == "sustained" else ("Likely technical fault" if anomaly.points > 0 else "No active issue")
ai_action_note = action_panel_ai_note(anomaly_data, anomaly.risk_level, possible_cause)

a1, a2, a3 = st.columns(3)
a1.metric("Suggested Inspection Time", inspection_time)
a2.metric("Priority", anomaly.risk_level)
a3.metric("Possible Cause", possible_cause)
st.write("🤖 Action Brief:")
st.write(ai_action_note)

st.markdown("### 4) Impact")
b1, b2, b3 = st.columns(3)
b1.metric("Estimated Energy Loss", f"{metrics.energy_loss_kwh:.2f} kWh")
b2.metric("Estimated Revenue Loss", f"₹ {revenue_loss:,.2f}")
b3.metric("Alert Threshold", f"{anomaly_threshold_pct:.1f}%")