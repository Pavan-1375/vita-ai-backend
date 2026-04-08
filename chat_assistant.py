from __future__ import annotations

import os
from typing import Iterable

import anthropic


def safety_disclaimer() -> str:
    return (
        "This assistant provides educational guidance only and is not a medical diagnosis. "
        "If symptoms are severe, worsening, or include emergency red flags, seek urgent care."
    )


def build_assistant_reply(
    user_message: str,
    predicted_disease: str,
    triage: str,
    precautions: Iterable[str],
    red_flags: Iterable[str],
) -> str:
    """
    Call Claude to generate a contextual health assistant reply.
    Falls back to a safe static message if the API call fails.
    """
    precautions_list = list(precautions)
    red_flag_list = list(red_flags)

    system_prompt = f"""You are a helpful, empathetic AI health assistant embedded inside a symptom checker app.

Current patient context:
- Predicted disease: {predicted_disease}
- Triage level: {triage.upper()}
- Precautions: {", ".join(precautions_list) if precautions_list else "None listed"}
- Emergency red flags detected: {", ".join(red_flag_list) if red_flag_list else "None"}

Your role:
- Answer the patient's follow-up health questions clearly and compassionately
- Refer back to the predicted disease and triage level where relevant
- Suggest practical next steps based on the precautions
- If red flags are present, always prioritise directing the patient to seek urgent care
- Never provide a definitive diagnosis — always remind the user this is educational guidance only
- Keep replies concise (3-5 sentences unless more detail is genuinely needed)
- Do not recommend specific prescription medications by name"""

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

    except anthropic.AuthenticationError:
        return (
            "API key missing or invalid. Add your ANTHROPIC_API_KEY to a "
            ".env file or your environment variables to enable the AI assistant."
        )
    except anthropic.RateLimitError:
        return (
            "The assistant is currently busy. Please wait a moment and try again."
        )
    except Exception as e:
        # Graceful fallback - never crash the Streamlit app
        return (
            f"I'm having trouble connecting to the AI right now ({type(e).__name__}). "
            f"Based on your results: predicted condition is {predicted_disease} "
            f"(triage: {triage.upper()}). "
            "Please follow the precautions listed above and consult a doctor if needed."
        )
