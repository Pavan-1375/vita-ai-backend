from __future__ import annotations

from dataclasses import dataclass
import os
import math
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SymptomModel:
    disease_df: pd.DataFrame
    precaution_df: pd.DataFrame
    symptom_columns: list[str]
    symptom_weight: dict[str, float]


def _csv_paths(base_dir: str) -> tuple[str, str]:
    disease_path = os.path.join(base_dir, "DiseaseAndSymptoms.csv")
    precaution_path = os.path.join(base_dir, "Disease precaution.csv")
    return disease_path, precaution_path


def load_model(base_dir: str) -> SymptomModel:
    disease_path, precaution_path = _csv_paths(base_dir)

    missing = [p for p in (disease_path, precaution_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required data file(s):\n"
            + "\n".join(f" - {p}" for p in missing)
            + "\n\nPlace both CSV files in the same folder as the app."
        )

    disease_df = pd.read_csv(disease_path)
    precaution_df = pd.read_csv(precaution_path)

    disease_df.columns = disease_df.columns.str.strip()
    precaution_df.columns = precaution_df.columns.str.strip()

    symptom_columns = [col for col in disease_df.columns if "Symptom" in col]
    for col in symptom_columns:
        disease_df[col] = disease_df[col].astype(str).str.strip().str.lower()

    disease_df["All_Symptoms"] = disease_df[symptom_columns].values.tolist()
    disease_df["All_Symptoms"] = disease_df["All_Symptoms"].apply(
        lambda x: [s for s in x if s != "nan"]
    )

    # Inverse-frequency weights: rarer symptoms are more informative.
    symptom_frequency: dict[str, int] = {}
    for symptom_list in disease_df["All_Symptoms"].tolist():
        for symptom in set(symptom_list):
            symptom_frequency[symptom] = symptom_frequency.get(symptom, 0) + 1

    total_diseases = max(len(disease_df), 1)
    symptom_weight: dict[str, float] = {}
    for symptom, freq in symptom_frequency.items():
        symptom_weight[symptom] = math.log((1 + total_diseases) / (1 + freq)) + 1.0

    return SymptomModel(
        disease_df=disease_df,
        precaution_df=precaution_df,
        symptom_columns=symptom_columns,
        symptom_weight=symptom_weight,
    )


def all_known_symptoms(model: SymptomModel) -> list[str]:
    symptoms: set[str] = set()
    for s_list in model.disease_df["All_Symptoms"].tolist():
        for s in s_list:
            if isinstance(s, str) and s and s != "nan":
                symptoms.add(s.strip().lower())
    return sorted(symptoms)


RED_FLAG_SYMPTOMS = {
    "chest pain",
    "shortness of breath",
    "breathlessness",
    "difficulty breathing",
    "severe headache",
    "fainting",
    "loss of consciousness",
    "confusion",
    "slurred speech",
    "weakness of one body side",
    "vomiting blood",
    "blood in sputum",
    "seizures",
}

DEFAULT_HOME_REMEDIES = [
    "Rest and stay well hydrated.",
    "Eat light, nutritious meals and avoid oily/spicy food.",
    "Use steam inhalation or warm fluids for mild throat/nasal symptoms.",
    "Monitor symptoms for 24-48 hours and seek care if worsening.",
]

DISEASE_HOME_REMEDIES = {
    "Common Cold": [
        "Drink warm fluids and take adequate rest.",
        "Use steam inhalation for nasal congestion.",
        "Use salt-water gargles for sore throat.",
        "Avoid cold/dust exposure and stay hydrated.",
    ],
    "COVID-19": [
        "Isolate and monitor temperature/oxygen if available.",
        "Drink fluids and take sufficient rest.",
        "Use mask to protect people around you.",
        "Contact doctor early if breathing worsens.",
    ],
    "Gastroenteritis": [
        "Take oral rehydration solution frequently.",
        "Eat bland food in small portions.",
        "Avoid dairy/spicy/oily meals temporarily.",
        "Watch for dehydration signs.",
    ],
    "Migraine": [
        "Rest in a dark and quiet room.",
        "Stay hydrated and avoid trigger foods.",
        "Reduce screen time during episodes.",
        "Use prescribed medication if advised by doctor.",
    ],
    "Allergy": [
        "Avoid known allergens and dust exposure.",
        "Keep rooms clean and ventilated.",
        "Use saline rinse for nasal symptoms.",
        "Use antihistamine only as prescribed.",
    ],
}

URGENT_ACTIONS = [
    "Call emergency services immediately.",
    "Do not self-medicate for severe chest pain, breathing issues, or fainting.",
    "Keep patient seated/lying safely and monitor breathing.",
    "Go to nearest emergency department without delay.",
]


def _dynamic_home_remedies(
    disease: str, matched_symptoms: list[str], precautions: list[str]
) -> list[str]:
    # Prefer explicit per-disease mapping when available.
    if disease in DISEASE_HOME_REMEDIES:
        return DISEASE_HOME_REMEDIES[disease]

    symptom_set = set(matched_symptoms)
    remedies: list[str] = []

    if {"cough", "sore throat", "runny nose", "nasal congestion"} & symptom_set:
        remedies.append("Use warm fluids and steam inhalation for throat/nasal relief.")
    if {"fever", "chills", "body pain"} & symptom_set:
        remedies.append("Take adequate rest, stay hydrated, and monitor temperature.")
    if {"diarrhoea", "vomiting", "nausea", "abdominal pain"} & symptom_set:
        remedies.append("Use oral rehydration and eat bland food in small portions.")
    if {"headache", "migraine", "light sensitivity"} & symptom_set:
        remedies.append("Rest in a calm, dark room and maintain hydration.")
    if {"itching", "rash", "skin redness"} & symptom_set:
        remedies.append("Keep skin clean and dry; avoid scratching or irritants.")
    if {"burning urination", "frequent urination"} & symptom_set:
        remedies.append("Increase water intake and maintain urinary hygiene.")

    # Convert selected precaution items into practical home steps.
    for p in precautions[:3]:
        p_low = str(p).lower()
        if "consult" in p_low or "doctor" in p_low or "hospital" in p_low:
            continue
        remedies.append(str(p).capitalize() + ".")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for item in remedies:
        if item not in seen:
            unique.append(item)
            seen.add(item)

    return unique[:4] if unique else DEFAULT_HOME_REMEDIES


def _score_candidate(
    input_symptoms: set[str], disease_symptoms: set[str], symptom_weight: dict[str, float]
) -> dict:
    overlap = input_symptoms & disease_symptoms
    raw_match = len(overlap)
    if raw_match == 0:
        return {"raw_match": 0, "weighted_score": 0.0, "coverage": 0.0, "overlap": []}

    weighted_hit = sum(symptom_weight.get(s, 1.0) for s in overlap)
    total_possible = sum(symptom_weight.get(s, 1.0) for s in disease_symptoms) or 1.0
    coverage = weighted_hit / total_possible
    return {
        "raw_match": raw_match,
        "weighted_score": weighted_hit,
        "coverage": coverage,
        "overlap": sorted(overlap),
    }


def triage_level(user_symptoms: Iterable[str], confidence: float) -> str:
    symptom_set = {s.strip().lower() for s in user_symptoms if str(s).strip()}
    if symptom_set & RED_FLAG_SYMPTOMS:
        return "high"
    if confidence >= 70:
        return "medium"
    return "low"


def predict_disease(model: SymptomModel, user_symptoms: Iterable[str]) -> dict:
    user_symptoms_list = [s.strip().lower() for s in user_symptoms if str(s).strip()]
    input_set = set(user_symptoms_list)
    red_flags = sorted(input_set & RED_FLAG_SYMPTOMS)

    # Safety-first override: if emergency red flags are present, avoid
    # returning reassuring/mild diagnoses as primary output.
    if red_flags:
        return {
            "Predicted Disease": "Emergency Warning",
            "Match Score": 0,
            "Precautions": [
                "Seek urgent medical care immediately.",
                "Call emergency services if chest pain or breathing issues are severe.",
                "Do not rely on this app for emergency diagnosis.",
            ],
            "Input Symptoms": user_symptoms_list,
            "Confidence": 100.0,
            "Top Predictions": [],
            "Triage": "high",
            "Red Flags": red_flags,
            "Home Remedies": [],
            "Urgent Actions": URGENT_ACTIONS,
        }

    ranked: list[dict] = []
    for _, row in model.disease_df.iterrows():
        disease = row["Disease"]
        disease_symptoms = set(row["All_Symptoms"])
        score = _score_candidate(input_set, disease_symptoms, model.symptom_weight)
        if score["raw_match"] == 0:
            continue
        ranked.append(
            {
                "disease": disease,
                "raw_match": score["raw_match"],
                "weighted_score": score["weighted_score"],
                "coverage": score["coverage"],
                "overlap": score["overlap"],
            }
        )

    ranked.sort(key=lambda x: (x["weighted_score"], x["raw_match"], x["coverage"]), reverse=True)

    if not ranked:
        return {
            "Predicted Disease": "Unknown",
            "Precautions": ["No strong symptom match found."],
            "Top Predictions": [],
            "Confidence": 0.0,
            "Triage": "low",
            "Red Flags": sorted(input_set & RED_FLAG_SYMPTOMS),
            "Home Remedies": DEFAULT_HOME_REMEDIES,
            "Urgent Actions": [],
        }

    best = ranked[0]
    best_match = best["disease"]
    precautions = model.precaution_df[model.precaution_df["Disease"] == best_match]
    if not precautions.empty:
        precaution_list = precautions.iloc[0][1:].dropna().tolist()
    else:
        precaution_list = ["No specific precautions found."]

    confidence = round(min(100.0, (best["coverage"] * 70) + (best["raw_match"] * 8)), 2)
    triage = triage_level(user_symptoms_list, confidence)
    top_predictions = [
        {
            "Disease": x["disease"],
            "Match Score": x["raw_match"],
            "Weighted Score": round(x["weighted_score"], 3),
            "Coverage": round(x["coverage"] * 100, 2),
            "Matched Symptoms": x["overlap"],
        }
        for x in ranked[:3]
    ]

    return {
        "Predicted Disease": best_match,
        "Match Score": best["raw_match"],
        "Precautions": precaution_list,
        "Input Symptoms": user_symptoms_list,
        "Confidence": confidence,
        "Top Predictions": top_predictions,
        "Triage": triage,
        "Red Flags": red_flags,
        "Home Remedies": _dynamic_home_remedies(
            best_match, best["overlap"], precaution_list
        ),
        "Urgent Actions": URGENT_ACTIONS if triage == "high" else [],
    }

