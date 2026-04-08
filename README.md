# Symptom Checker (Frontend)

## Setup

From this folder:

```powershell
python -m pip install -r requirements.txt
```

## Run (web UI)

```powershell
streamlit run app.py
```

## New Hackathon Features

- Top-3 weighted predictions (not only one best match)
- Triage label (`LOW`, `MEDIUM`, `HIGH`) + emergency red-flag detection
- Voice input (microphone) to text
- Voice output (text-to-speech) for result summary
- In-app chat assistant for follow-up guidance
- Safety filter toggle for advice quality
- Downloadable visit report (`.txt`)

## Notes

- Voice input uses Google's speech recognition backend (internet needed).
- Voice output uses gTTS (internet needed).
- This is not a medical diagnosis tool. Always consult a licensed professional.

## 60-Second Demo Script

1. Open app and show "Safety filter: ON".
2. Enter mild symptoms: `cough, runny nose` and click **Check**.
3. Show top-3 predictions + precautions + home remedies.
4. Click **Download Report (TXT)** to show export.
5. Enter emergency symptom: `chest pain`.
6. Show emergency warning and urgent actions.

## Run (CLI)

Interactive:

```powershell
python symptom.py
```

Non-interactive (pass symptoms):

```powershell
python symptom.py "fever, cough, headache"
```

