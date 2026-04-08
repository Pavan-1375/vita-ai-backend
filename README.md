# YouOkay — Backend

Core engine for symptom prediction, triage, and safety checks.

### Features
- Matches user symptoms against 400+ conditions using CSV data
- Weighted symptom scoring + top-3 results
- Triage classification (LOW / MEDIUM / HIGH)
- Red-flag emergency detection
- Contextual chat assistant
- Precaution & remedy lookup
- Voice-ready components

### Main Files
- `symptom_core.py` — Core prediction logic
- `app.py` — Streamlit UI
- `chat_assistant.py` — AI chat
- `api.py` / `server.py` — API endpoints

### Tech Stack
- Python 3, Pandas, Streamlit
- gTTS + SpeechRecognition for voice
- CSV knowledge base (`DiseaseAndSymptoms.csv`)

### How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
