import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _safe_chat(prompt: str):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception:
        return None


def _fallback_insights(data: dict) -> dict:
    loss_pct = float(data.get("loss_pct", 0.0))
    deviation = float(data.get("deviation_from_normal", 0.0))
    duration = float(data.get("duration_hours", 0.0))
    trend = str(data.get("trend", "stable"))
    risk = str(data.get("risk_level", "LOW"))

    if risk == "HIGH" or loss_pct >= 22:
        severity = "HIGH"
        recommended_action = "Dispatch inspection team immediately, isolate suspect section, and audit meter integrity."
    elif risk == "MEDIUM" or loss_pct >= 12:
        severity = "MEDIUM"
        recommended_action = "Schedule same-day site inspection, validate CT/PT wiring, and reconcile transformer vs meter logs."
    else:
        severity = "LOW"
        recommended_action = "Continue monitoring, run preventive checks, and review trend in next cycle."

    if trend == "sustained" and deviation > 0:
        root_cause = "Sustained abnormal loss pattern suggests possible theft or persistent metering mismatch."
    elif deviation > 0:
        root_cause = "Short-duration anomaly indicates likely technical disturbance rather than persistent theft."
    else:
        root_cause = "No significant abnormality detected against normal operating baseline."

    explanation = (
        f"Feeder {data.get('feeder_id')} shows {loss_pct:.2f}% loss "
        f"({deviation:+.2f}% vs 10% baseline) over {duration:.1f}h with {trend} trend."
    )

    summary = f"{severity} risk: {trend} anomaly with {loss_pct:.2f}% loss; operational review recommended."

    return {
        "explanation": explanation,
        "root_cause_analysis": root_cause,
        "severity": severity,
        "recommended_action": recommended_action,
        "summary": summary,
        "ai_source": "fallback",
    }


def generate_insights(data: dict) -> dict:
    prompt = f"""
You are an expert electricity grid analyst.
Analyze the following real feeder anomaly context and respond as strict JSON.

Context:
{json.dumps(data, indent=2)}

Reason step-by-step internally using:
1) Current loss % vs normal baseline (10%)
2) Trend type (spike/sustained)
3) Duration and recent loss readings
4) Whether pattern is more likely theft or fault

Return only JSON with keys:
- explanation
- root_cause_analysis
- severity (LOW/MEDIUM/HIGH)
- recommended_action
- summary

Constraints:
- Be specific to the provided numbers.
- Keep explanation concise and actionable.
"""

    raw = _safe_chat(prompt)
    if not raw:
        return _fallback_insights(data)

    try:
        parsed = json.loads(raw)
        return {
            "explanation": parsed.get("explanation", "No explanation generated."),
            "root_cause_analysis": parsed.get("root_cause_analysis", "No root cause analysis generated."),
            "severity": parsed.get("severity", data.get("risk_level", "MEDIUM")),
            "recommended_action": parsed.get("recommended_action", "Inspect feeder and validate meter integrity."),
            "summary": parsed.get("summary", "Anomaly detected; inspection advised."),
            "ai_source": "openai",
        }
    except Exception:
        return _fallback_insights(data)


def explain_anomaly(data):
    return generate_insights(data)["explanation"]


def recommend_action(data):
    return generate_insights(data)["recommended_action"]


def short_summary(data):
    return generate_insights(data)["summary"]


def ask_grid(question, anomaly_data, feeder_snapshot):
    prompt = f"""
You are a smart electricity grid assistant.

Feeder anomaly context:
{anomaly_data}

Latest feeder snapshot:
{feeder_snapshot}

User question:
{question}

Answer with practical, field-oriented guidance in 4-6 lines.
Use the recent rows to justify your answer.
"""
    out = _safe_chat(prompt)
    if out:
        return out
    return (
        f"Using rule-based assistant mode: feeder {anomaly_data.get('feeder_id')} is at "
        f"{anomaly_data.get('loss_pct')}% loss with {anomaly_data.get('trend')} trend. "
        f"For '{question}', recommended next step is to inspect recent anomaly intervals first."
    )


def action_panel_ai_note(anomaly_data, rule_priority, likely_cause):
    prompt = f"""
You are a grid operations lead.

Data:
{anomaly_data}

Rule priority: {rule_priority}
Likely cause: {likely_cause}

Return 3 concise lines:
1) Why this priority is justified
2) Immediate first action
3) Field team instruction
"""
    out = _safe_chat(prompt)
    if out:
        return out
    return (
        f"Priority {rule_priority} is consistent with current feeder risk. "
        f"Immediate action: verify meter-line balance and inspect suspected segment. "
        f"Likely cause to validate: {likely_cause}."
    )