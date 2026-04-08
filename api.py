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
    version="1.2.0",
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
    "chest discomfort": "chest_pain",
    "pressure in chest": "chest_pain",
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
        
        # Fix aliases
        s = SYMPTOM_ALIASES.get(s, s)
        
        # CRITICAL: Convert spaces to underscores to match the CSV database
        s = s.replace(" ", "_")
        
        # If it's an exact match in our database, keep it
        if s in known_symptoms:
            cleaned.append(s)
            continue
            
        # SMART EXTRACT: If user types "severe one-sided headache", find "headache"
        matched = False
        for known in known_symptoms:
            if known in s or s in known:
                cleaned.append(known)
                matched = True
                break
                
        if not matched:
            cleaned.append(s)

    # Keep order, remove duplicates
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
    # Removed "Note" field so UI stays clean
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
    }


def _apply_safety_overrides(signals: List[str], result: dict) -> dict:
    # Strong emergency override for chest pain/cardiac pattern
    if _has_any(signals, ["heart attack", "heartattack", "cardiac", "angina", "chest_pain", "chest pain"]):
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

    # Nosebleed-specific override
    if _has_any(signals, ["nosebleed", "bloody nose", "bleeding nose", "nose_bleeding"]):
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
            }

        confidence = float(result.get("Confidence", 0) or 0)
    disease = str(result.get("Predicted Disease", "")).strip().lower()
    
    # If no disease was found at all, use fallback
    if not disease or disease == "unknown":
        return _build_low_signal_fallback(signals)

    # If confidence is low BUT we found a real disease, keep the real disease 
    # and its precautions! Just force Triage to "low" for safety.
    if confidence < 35:
        result["Confidence"] = max(confidence, 30) # Clean up the number slightly
        result["Triage"] = "low"
        result["Input Symptoms"] = signals
        return result

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
    # Convert underscores to spaces so the React UI looks clean to the user
    clean_symptoms = [s.replace("_", " ").strip() for s in known_symptoms]
    return {"symptoms": clean_symptoms}


@app.post("/predict")
def predict(req: SymptomRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Symptom model not loaded.")

    signals = _normalize_symptoms(req.symptoms)
    if not signals:
        raise HTTPException(status_code=400, detail="Provide at least one symptom.")

    result = predict_disease(model, signals)
    return _apply_safety_overrides(signals, result)


@app.post("/chat")
def chat(req: ChatRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {
            "reply": "AI chat is temporarily unavailable. Follow the precautions listed above, and consult a doctor if symptoms worsen."
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
            "reply": "AI assistant is currently offline. Follow the precautions listed in your analysis."
        }

    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # UPGRADED PROMPT: Forces Claude to explain precautions clearly like a real doctor
        enhanced_prompt = req.system_prompt + """

STRICT INSTRUCTIONS FOR PRECAUTIONS & REMEDIES:
1. When the user asks what to do, or asks about precautions/remedies, you MUST use the exact precautions and home remedies provided in the analysis result above.
2. Do NOT invent new precautions. Only explain the ones from the analysis.
3. Format them clearly using bullet points.
4. Explain briefly WHY each precaution helps (e.g., "Hydrate: This helps flush out toxins and keeps your body recovering faster").
5. Keep the tone empathetic, clear, and professional. Do not sound like a robot."""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300, # Kept short so replies are fast and punchy
            system=enhanced_prompt,
            messages=req.messages,
        )
        return {"reply": response.content[0].text}
    except Exception as e:
        message = str(e)
        lower = message.lower()
        if "credit balance" in lower or "billing" in lower or "rate limit" in lower:
            return {
                "reply": "Premium AI is temporarily unavailable. Stick to the precautions listed in your analysis."
            }
        raise HTTPException(status_code=500, detail=message)
