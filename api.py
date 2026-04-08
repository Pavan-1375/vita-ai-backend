from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

from symptom_core import load_model, predict_disease, all_known_symptoms
from chat_assistant import build_assistant_reply

load_dotenv()

app = FastAPI(
    title="VitalAI Health API",
    description="Symptom checker + AI chat assistant backend",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)

try:
    model = load_model(BASE_DIR)
    known_symptoms = all_known_symptoms(model)
    print("Symptom model loaded successfully")
except FileNotFoundError as e:
    print(f"Model load failed: {e}")
    model = None
    known_symptoms = []


class SymptomRequest(BaseModel):
    symptoms: List[str]


class ChatRequest(BaseModel):
    message: str
    predicted_disease: str
    triage: str
    precautions: List[str]
    red_flags: List[str]
    conversation_history: List[dict] = []


class ClaudeRequest(BaseModel):
    messages: List[dict]
    system_prompt: str


SYMPTOM_ALIASES = {
    "nose bleeding": "nosebleed",
    "bleeding nose": "nosebleed",
    "bloody nose": "nosebleed",
    "chest discomfort": "chest pain",
    "pressure in chest": "chest pain",
    "stuffy nose": "congestion",
    "runny nose": "congestion",
    "throwing up": "vomiting",
}


def _normalize_symptoms(symptoms: List[str]) -> List[str]:
    cleaned: List[str] = []
    for raw in symptoms:
        s = str(raw).strip().lower()
        if not s:
            continue
        s = SYMPTOM_ALIASES.get(s, s)
        cleaned.append(s)
    # keep order, remove duplicates
    seen = set()
    out: List[str] = []
    for s in cleaned:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _has_any(signals: List[str], keywords: List[str]) -> bool:
    for sig in signals:
        for key in keywords:
            if sig == key or key in sig or sig in key:
                return True
    return False


def _build_low_signal_fallback(signals: List[str]) -> dict:
    return {
        "Predicted Disease": "Non-specific Symptom Cluster",
        "Confidence": 52,
        "Triage": "low",
        "Precautions": [
            "Track symptom intensity every 6-8 hours",
            "Hydrate and rest",
            "Seek clinician review if symptoms persist or worsen",
        ],
        "Home Remedies": [
            "Light meals",
            "Sleep hygiene",
            "Avoid self-medicating with unfamiliar drugs",
        ],
        "Urgent Actions": [],
        "Input Symptoms": signals,
        "Note": "Low confidence safety fallback applied.",
    }


def _apply_safety_overrides(signals: List[str], result: dict) -> dict:
    # Strong emergency override for chest pain/cardiac pattern
    if _has_any(signals, ["heart attack", "heartattack", "cardiac", "angina", "chest pain"]):
        return {
            "Predicted Disease": "Possible cardiac emergency pattern",
            "Confidence": 92,
            "Triage": "high",
            "Precautions": [
                "Call emergency services immediately",
                "Stop activity and sit or lie down safely",
                "Do not delay in-person emergency care",
            ],
            "Home Remedies": ["No home remedy for suspected heart emergency."],
            "Urgent Actions": ["Go to ER now."],
            "Input Symptoms": signals,
            "Top Predictions": result.get("Top Predictions", []),
        }

    # Nosebleed-specific override to avoid unsafe neurological over-call when signal is weak
    if _has_any(signals, ["nosebleed", "bloody nose", "bleeding nose"]):
        confidence = float(result.get("Confidence", 0) or 0)
        if confidence < 45:
            return {
                "Predicted Disease": "Likely Epistaxis (Nosebleed Pattern)",
                "Confidence": 76,
                "Triage": "medium",
                "Precautions": [
                    "Sit upright and lean slightly forward",
                    "Pinch soft part of nose for 10-15 minutes continuously",
                    "Apply cold compress over nose bridge",
                    "Do not tilt head back",
                ],
                "Home Remedies": [
                    "Hydrate and keep room air humidified",
                    "Use saline gel for dry nostrils",
                ],
                "Urgent Actions": [
                    "Seek urgent care if bleeding lasts more than 20 minutes",
                    "Seek urgent care if dizziness, fainting, or heavy bleeding occurs",
                ],
                "Input Symptoms": signals,
                "Top Predictions": result.get("Top Predictions", []),
                "Note": "Safety override applied due low-confidence mismatch.",
            }

    confidence = float(result.get("Confidence", 0) or 0)
    disease = str(result.get("Predicted Disease", "")).strip().lower()
    if confidence < 35 or not disease or disease == "unknown":
        return _build_low_signal_fallback(signals)

    result["Input Symptoms"] = signals
    return result


@app.get("/")
def root():
    return {
        "status": "VitalAI API is running",
        "endpoints": ["/symptoms", "/predict", "/chat", "/claude", "/docs"],
    }


@app.get("/symptoms")
def get_symptoms():
    return {"symptoms": known_symptoms}


@app.post("/predict")
def predict(req: SymptomRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Symptom model not loaded. Check CSV files.")

    signals = _normalize_symptoms(req.symptoms)
    if not signals:
        raise HTTPException(status_code=400, detail="Provide at least one symptom.")

    result = predict_disease(model, signals)
    return _apply_safety_overrides(signals, result)


@app.post("/chat")
def chat(req: ChatRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        # Keep app usable even without paid key.
        return {
            "reply": "AI chat is temporarily unavailable right now. Follow your precautions, monitor red flags, and consult a doctor if symptoms worsen."
        }

    reply = build_assistant_reply(
        user_message=req.message,
        predicted_disease=req.predicted_disease,
        triage=req.triage,
        precautions=req.precautions,
        red_flags=req.red_flags,
    )
    return {"reply": reply}


@app.post("/claude")
def claude_direct(req: ClaudeRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {
            "reply": "AI assistant is currently offline. Use symptom checker precautions for now and try again later."
        }

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=req.system_prompt,
            messages=req.messages,
        )
        return {"reply": response.content[0].text}
    except Exception as e:
        message = str(e)
        lower = message.lower()
        if "credit balance is too low" in lower or "billing" in lower or "rate limit" in lower:
            return {
                "reply": "Premium AI is temporarily unavailable due to billing/rate limits. Basic safety guidance is still active."
            }
        raise HTTPException(status_code=500, detail=message)