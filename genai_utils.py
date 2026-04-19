import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _safe_chat(prompt: str) -> str:
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as exc:
        return f"GenAI unavailable right now: {exc}"


def explain_anomaly(data):
    prompt = f"""
You are an electricity grid analyst.

Feeder: {data['feeder_id']}
Loss: {data['loss_pct']}%
Duration: {data['duration_hours']} hours
Trend: {data.get('trend','unknown')}

Explain clearly:
- what is happening
- why it is suspicious

Keep it under 4 lines.
"""
    return _safe_chat(prompt)


def recommend_action(data):
    prompt = f"""
You are a grid operations expert.

Feeder: {data['feeder_id']}
Loss: {data['loss_pct']}%
Duration: {data['duration_hours']} hours

Give:
1. Action
2. Priority (Low/Medium/High)
3. Likely cause

Be concise.
"""
    return _safe_chat(prompt)


def short_summary(data):
    prompt = f"""
Summarize this anomaly in 1 line:

Feeder: {data['feeder_id']}
Loss: {data['loss_pct']}%
"""
    return _safe_chat(prompt)


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
"""
    return _safe_chat(prompt)


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
    return _safe_chat(prompt)