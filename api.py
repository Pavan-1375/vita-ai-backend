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
    version="1.3.0",
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
    "burning urination": "burning_micturition",
    "burning urine": "burning_micturition",
}


def _normalize_symptoms(symptoms: List[str]) -> List[str]:
    cleaned: List[str] = []
    for raw in symptoms:
        s = str(raw).strip().lower()
        if not s:
            continue
        s = SYMPTOM_ALIASES.get(s, s)
        s = s.replace(" ", "_")
        
        if s in known_symptoms:
            cleaned.append(s)
            continue
            
        matched = False
        for known in known_symptoms:
            if known in s or s in known:
                cleaned.append(known)
                matched = True
                break
                
        if not matched:
            cleaned.append(s)

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
    }


def _apply_safety_overrides(signals: List[str], result: dict) -> dict:
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

       # Get final confidence and disease
    confidence = float(result.get("Confidence", 0) or 0)
    disease = str(result.get("Predicted Disease", "")).strip().lower()
    
    # If no disease was found at all, use fallback
    if not disease or disease == "unknown":
        return _build_low_signal_fallback(signals)

    # CRITICAL SAFETY CHECK: Never show severe diseases (Paralysis, AIDS, etc.) 
    # if confidence is low. A user typing "headache" should NEVER see "Paralysis".
    severe_keywords = ["paralysis", "aids", "heart attack", "tuberculosis", "hepatitis", "dengue", "malaria", "pneumonia", "typhoid"]
    is_severe_disease = any(sev in disease for sev in severe_keywords)
    
        if confidence < 40 and is_severe_disease:
        # SMART FALLBACK: If user types 1 common symptom, give a real answer, not "Non-specific"
        single_symptom = signals[0] if len(signals) == 1 else ""
        
        if "headache" in single_symptom:
            return {
                "Predicted Disease": "Tension Headache or Migraine",
                "Confidence": 45,
                "Triage": "low",
                "Precautions": ["Rest in a dark, quiet room", "Stay hydrated", "Limit screen time", "Monitor for 24 hours"],
                "Home Remedies": ["Apply a cold compress to your forehead.", "Drink plenty of water.", "Take a short nap."],
                "Urgent Actions": [],
                "Input Symptoms": signals,
                "Top Predictions": result.get("Top Predictions", [])
            }
        elif "fever" in single_symptom:
            return {
                "Predicted Disease": "Viral Fever",
                "Confidence": 45,
                "Triage": "medium",
                "Precautions": ["Monitor temperature regularly", "Stay hydrated", "Rest", "Consult doctor if it persists beyond 3 days"],
                "Home Remedies": ["Apply a lukewarm sponge bath.", "Drink plenty of electrolytes or ORS.", "Rest under a light blanket."],
                "Urgent Actions": ["Seek immediate care if temperature crosses 103°F (39.4°C)."],
                "Input Symptoms": signals,
                "Top Predictions": result.get("Top Predictions", [])
            }
        elif "cough" in single_symptom:
            return {
                "Predicted Disease": "Upper Respiratory Infection",
                "Confidence": 45,
                "Triage": "low",
                "Precautions": ["Stay hydrated", "Avoid cold or dusty environments", "Monitor for 48 hours"],
                "Home Remedies": ["Drink warm turmeric milk.", "Sip on hot ginger tea with honey.", "Use a steam vaporizer to clear airways."],
                "Urgent Actions": [],
                "Input Symptoms": signals,
                "Top Predictions": result.get("Top Predictions", [])
            }
        else:
            # Only show "Non-specific" for weird inputs, not common words
            return _build_low_signal_fallback(signals)

def _get_safe_dataset_override(signals: List[str]) -> dict | None:
    """Forces generic standalone symptoms to match a safe disease with exact CSV precautions."""
    if len(signals) != 1:
        return None # Only triggers if user types exactly ONE symptom
    
    symptom = signals[0].lower()

    if symptom == "headache":
        return {
            "Predicted Disease": "Migraine",
            "Confidence": 45,
            "Triage": "low",
            "Precautions": ["Meditation", "Reduce stress", "Use polaroid glasses in sun", "Consult doctor"],
            "Home Remedies": ["Apply a cold compress to your forehead.", "Rest in a dark, quiet room.", "Drink plenty of water to stay hydrated."],
            "Urgent Actions": [],
            "Red Flags": [],
            "Top Predictions": [{"Disease": "Migraine", "Match Score": 1, "Weighted Score": 4.5, "Coverage": 20.0, "Matched Symptoms": ["headache"]}]
        }
    
    elif symptom == "cough":
        return {
            "Predicted Disease": "Common Cold",
            "Confidence": 45,
            "Triage": "low",
            "Precautions": ["Drink vitamin C rich drinks", "Take vapour", "Avoid cold food", "Keep fever in check"],
            "Home Remedies": ["Gargle with warm salt water.", "Drink hot honey and lemon tea.", "Use a humidifier or steam inhalation."],
            "Urgent Actions": [],
            "Red Flags": [],
            "Top Predictions": [{"Disease": "Common Cold", "Match Score": 1, "Weighted Score": 4.5, "Coverage": 20.0, "Matched Symptoms": ["cough"]}]
        }
        
    elif symptom == "vomiting":
        return {
            "Predicted Disease": "Gastroenteritis",
            "Confidence": 45,
            "Triage": "medium",
            "Precautions": ["Stop eating solid food for a while", "Try taking small sips of water", "Rest", "Ease back into eating"],
            "Home Remedies": ["Take small sips of water or ice chips.", "Stick to bland foods like crackers or toast.", "Rest and avoid solid foods until vomiting stops."],
            "Urgent Actions": [],
            "Red Flags": [],
            "Top Predictions": [{"Disease": "Gastroenteritis", "Match Score": 1, "Weighted Score": 4.5, "Coverage": 20.0, "Matched Symptoms": ["vomiting"]}]
        }
        
    elif symptom in ["skin_rash", "itching"]:
        return {
            "Predicted Disease": "Fungal infection",
            "Confidence": 45,
            "Triage": "low",
            "Precautions": ["Bath twice", "Use detol or neem in bathing water", "Keep infected area dry", "Use clean cloths"],
            "Home Remedies": ["Apply a cool, damp cloth to the affected area.", "Use an unscented moisturizer.", "Avoid scratching the area."],
            "Urgent Actions": [],
            "Red Flags": [],
            "Top Predictions": [{"Disease": "Fungal infection", "Match Score": 1, "Weighted Score": 4.5, "Coverage": 20.0, "Matched Symptoms": [symptom]}]
        }

    return None
    
@app.get("/")
def root():
    return {
        "status": "VitalAI API is running",
        "endpoints": ["/symptoms", "/predict", "/chat", "/claude", "/docs"],
    }


@app.get("/symptoms")
def get_symptoms():
    clean_symptoms = [s.replace("_", " ").strip() for s in known_symptoms]
    return {"symptoms": clean_symptoms}


@app.post("/predict")
def predict(req: SymptomRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Symptom model not loaded.")

        signals = _normalize_symptoms(req.symptoms)
    if not signals:
        raise HTTPException(status_code=400, detail="Provide at least one symptom.")

    try:
        # OVERRIDE: If user types a single generic symptom (like 'headache'), 
        # use the exact dataset match instead of letting the math guess.
        safe_override = _get_safe_dataset_override(signals)
        if safe_override:
            final_result = safe_override
        else:
            result = predict_disease(model, signals)
            final_result = _apply_safety_overrides(signals, result)

        top_preds = final_result.get("Top Predictions", [])
        if top_preds:
            seen_diseases = set()
            unique_preds = []
            for p in top_preds:
                disease_name = p.get("Disease", "")
                if disease_name not in seen_diseases:
                    seen_diseases.add(disease_name)
                    unique_preds.append(p)
            final_result["Top Predictions"] = unique_preds

        if "Precautions" in final_result and isinstance(final_result["Precautions"], list):
            final_result["Precautions"] = [item.capitalize() for item in final_result["Precautions"] if item]

        # AI HOME REMEDIES GENERATOR
        triage_level = str(final_result.get("Triage", "")).lower()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        disease_name = str(final_result.get("Predicted Disease", "")).lower()
        
        if triage_level == "high":
            final_result["Home Remedies"] = ["No home remedy recommended. Seek immediate medical attention."]
        
        elif "gerd" in disease_name or "acidity" in disease_name:
            final_result["Home Remedies"] = ["Drink cold milk to soothe the stomach lining.", "Chew sugar-free gum to reduce acid reflux.", "Sip ginger tea to ease digestion."]
        elif "migraine" in disease_name or "headache" in disease_name:
            final_result["Home Remedies"] = ["Apply a cold compress to your forehead.", "Rest in a dark, quiet room.", "Drink plenty of water to stay hydrated."]
        elif "common cold" in disease_name or "congestion" in disease_name:
            final_result["Home Remedies"] = ["Gargle with warm salt water.", "Drink hot honey and lemon tea.", "Use a humidifier or steam inhalation."]
        elif "fever" in disease_name or "malaria" in disease_name or "typhoid" in disease_name:
            final_result["Home Remedies"] = ["Apply a lukewarm sponge bath.", "Drink plenty of electrolytes or ORS.", "Rest under a light blanket."]
        elif "cough" in disease_name:
            final_result["Home Remedies"] = ["Drink warm turmeric milk.", "Sip on hot ginger tea with honey.", "Use a steam vaporizer to clear airways."]
        elif "skin_rash" in disease_name or "fungal" in disease_name or "acne" in disease_name:
            final_result["Home Remedies"] = ["Apply a cool, damp cloth to the affected area.", "Use an unscented moisturizer.", "Avoid scratching the area."]
        elif "vomiting" in disease_name or "gastroenteritis" in disease_name:
            final_result["Home Remedies"] = ["Take small sips of water or ice chips.", "Stick to bland foods like crackers or toast.", "Rest and avoid solid foods until vomiting stops."]
            
        elif api_key:
            try:
                import json
                import re
                client = anthropic.Anthropic(api_key=api_key)
                precautions = final_result.get("Precautions", [])
                
                remedy_prompt = f"""Generate exactly 3 simple, safe home remedies for a patient with '{disease_name}'. 
Do NOT just repeat these precautions: {precautions}. 
Suggest actual home care (e.g., ginger tea, warm compress, hydration).
Respond ONLY with a raw JSON array of 3 strings. No markdown, no extra text."""
                
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=100,
                    messages=[{"role": "user", "content": remedy_prompt}]
                )
                
                if response.content and len(response.content) > 0:
                    raw_text = response.content[0].text.strip()
                    raw_text = re.sub(r'^```json\s*|\s*```$', '', raw_text, flags=re.MULTILINE).strip()
                    try:
                        ai_remedies = json.loads(raw_text)
                        if isinstance(ai_remedies, list) and len(ai_remedies) > 0:
                            final_result["Home Remedies"] = [r.capitalize() for r in ai_remedies]
                    except json.JSONDecodeError:
                        pass 
            except Exception:
                pass 

        return final_result

    except Exception as e:
        return {
            "Predicted Disease": "Processing Error",
            "Confidence": 0,
            "Triage": "low",
            "Precautions": ["Please try analyzing your symptoms again."],
            "Home Remedies": [],
            "Urgent Actions": [],
            "Input Symptoms": signals
        }


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
        
        enhanced_prompt = req.system_prompt + """

STRICT INSTRUCTIONS FOR PRECAUTIONS & REMEDIES:
1. When the user asks what to do, or asks about precautions/remedies, you MUST use the exact precautions and home remedies provided in the analysis result above.
2. Do NOT invent new precautions. Only explain the ones from the analysis.
3. Format them clearly using bullet points.
4. Explain briefly WHY each precaution helps (e.g., "Hydrate: This helps flush out toxins and keeps your body recovering faster").
5. Keep the tone empathetic, clear, and professional. Do not sound like a robot."""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
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
