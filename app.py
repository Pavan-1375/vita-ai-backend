from __future__ import annotations

from io import BytesIO
import os

from dotenv import load_dotenv
load_dotenv()

from gTTS import gTTS
import speech_recognition as sr
import streamlit as st

from chat_assistant import build_assistant_reply, safety_disclaimer
from symptom_core import all_known_symptoms, load_model, predict_disease


BASE_DIR = os.path.dirname(__file__)


@st.cache_data(show_spinner=False)
def _load():
    model = load_model(BASE_DIR)
    symptoms = all_known_symptoms(model)
    return model, symptoms


def _parse_free_text(text: str) -> list[str]:
    parts = [p.strip().lower() for p in text.split(",")]
    return [p for p in parts if p]


def _filter_safe_advice(items: list[str]) -> list[str]:
    blocked_terms = {
        "self-medicate",
        "self medication",
        "without prescription",
        "unverified",
        "herbal cure",
        "alternative cure",
    }
    safe_items: list[str] = []
    for item in items:
        text = item.lower()
        if any(term in text for term in blocked_terms):
            continue
        safe_items.append(item)
    return safe_items


def _tts_audio_bytes(text: str) -> bytes | None:
    try:
        audio_fp = BytesIO()
        gTTS(text=text, lang="en").write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read()
    except Exception:
        return None


def _build_report_text(result: dict) -> str:
    disease = str(result.get("Predicted Disease", "Unknown"))
    triage = str(result.get("Triage", "low")).upper()
    confidence = float(result.get("Confidence", 0))
    match_score = int(result.get("Match Score", 0))
    input_symptoms = result.get("Input Symptoms", []) or []
    precautions = result.get("Precautions", []) or []
    remedies = result.get("Home Remedies", []) or []
    urgent_actions = result.get("Urgent Actions", []) or []

    lines = [
        "AI Symptom Checker - Visit Summary",
        "=" * 36,
        f"Predicted Disease: {disease}",
        f"Triage: {triage}",
        f"Confidence: {confidence}%",
        f"Match Score: {match_score}",
        "",
        "Input Symptoms:",
    ]
    lines.extend([f"- {s}" for s in input_symptoms] or ["- None"])
    lines.append("")
    lines.append("Precautions:")
    lines.extend([f"- {p}" for p in precautions] or ["- None"])
    lines.append("")
    lines.append("Home Remedies:")
    lines.extend([f"- {r}" for r in remedies] or ["- None"])
    if urgent_actions:
        lines.append("")
        lines.append("Urgent Actions:")
        lines.extend([f"- {u}" for u in urgent_actions])
    lines.append("")
    lines.append("Note: This tool is not a medical diagnosis.")
    return "\n".join(lines)


def _transcribe_audio(uploaded_audio) -> str:
    if uploaded_audio is None:
        return ""
    recognizer = sr.Recognizer()
    with sr.AudioFile(BytesIO(uploaded_audio.read())) as source:
        audio_data = recognizer.record(source)
    try:
        recognize_google = getattr(recognizer, "recognize_google", None)
        if callable(recognize_google):
            return str(recognize_google(audio_data))
        return ""
    except Exception:
        return ""


def _handle_chat_message(
    user_message: str, disease: str, triage: str, precautions: list[str], red_flags: list[str]
) -> None:
    clean_message = user_message.strip()
    if not clean_message:
        return
    st.session_state.chat_history.append(("user", clean_message))

    with st.spinner("Assistant is thinking..."):
        reply = build_assistant_reply(
            user_message=clean_message,
            predicted_disease=disease,
            triage=triage,
            precautions=precautions,
            red_flags=red_flags,
        )

    st.session_state.chat_history.append(("assistant", reply))
    st.session_state.last_chat_reply_audio = _tts_audio_bytes(reply)


# ── API key check ──────────────────────────────────────────────────────────────
if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error(
        "ANTHROPIC_API_KEY not found. "
        "Create a .env file in this folder with: ANTHROPIC_API_KEY=sk-ant-your-key-here"
    )
    st.stop()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Symptom Checker", page_icon="🩺", layout="centered")

st.title("AI Symptom Checker")
st.caption("Hackathon edition: top-3 prediction, triage, voice, and chat assistant.")
st.info(safety_disclaimer())
safe_only = st.toggle("Show only medically safe advice", value=True)
if safe_only:
    st.success("Safety filter: ON")
else:
    st.warning("Safety filter: OFF")

try:
    model, known_symptoms = _load()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

selected = st.multiselect(
    "Pick symptoms (optional)",
    options=known_symptoms,
    placeholder="Start typing a symptom…",
)

free_text = st.text_input(
    "Or type symptoms separated by commas (optional)",
    placeholder="e.g. fever, cough, headache",
)

voice_audio = st.audio_input("Voice input (optional)")
voice_text = _transcribe_audio(voice_audio) if voice_audio is not None else ""
if voice_text:
    st.success(f"Voice recognized: {voice_text}")

user_symptoms = sorted(set([*selected, *_parse_free_text(free_text), *_parse_free_text(voice_text)]))

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    run = st.button("Check", type="primary", use_container_width=True)
with col2:
    speak = st.button("Speak Result", use_container_width=True)
with col3:
    st.write("")
    st.write(f"Symptoms entered: **{len(user_symptoms)}**")

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_chat_reply_audio" not in st.session_state:
    st.session_state.last_chat_reply_audio = None

if run and user_symptoms:
    with st.spinner("Analysing symptoms..."):
        st.session_state.last_result = predict_disease(model, user_symptoms)

if run:
    if not user_symptoms:
        st.warning("Please enter at least one symptom.")
        st.stop()

result = st.session_state.last_result

if result:
    disease = result.get("Predicted Disease", "Unknown")
    st.subheader("Result")
    st.markdown(
        f"""
<div style="padding:12px;border-radius:10px;border:1px solid #2f80ed;background:#f7fbff;">
  <div style="font-size:14px;color:#555;">Predicted Disease</div>
  <div style="font-size:24px;font-weight:700;color:#1b4f72;">{disease}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    triage = str(result.get("Triage", "low")).lower()
    confidence = float(result.get("Confidence", 0))
    score = int(result.get("Match Score", 0))
    red_flags = result.get("Red Flags", []) or []

    c1, c2, c3 = st.columns(3)
    c1.metric("Match score", score)
    c2.metric("Confidence", f"{confidence}%")
    c3.metric("Triage", triage.upper())

    if red_flags:
        st.error(
            "Emergency red flags detected: "
            + ", ".join(red_flags)
            + ". Please seek urgent medical care."
        )

    st.subheader("Top 3 Predictions")
    st.dataframe(result.get("Top Predictions", []), use_container_width=True)

    report_text = _build_report_text(result)
    st.download_button(
        "Download Report (TXT)",
        data=report_text,
        file_name="symptom_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.subheader("Precautions")
    precautions = result.get("Precautions", []) or []
    for p in precautions:
        st.write(f"- {p}")

    st.subheader("Home Remedies (Mild Cases)")
    remedies = result.get("Home Remedies", []) or []
    if safe_only:
        remedies = _filter_safe_advice(remedies)
    if remedies:
        for r in remedies:
            st.write(f"- {r}")
    else:
        st.write("- No home remedies suggested for this case.")

    urgent_actions = result.get("Urgent Actions", []) or []
    if urgent_actions:
        st.subheader("Urgent Actions")
        st.error("This case may need urgent medical attention.")
        for action in urgent_actions:
            st.write(f"- {action}")

    if speak:
        speak_text = (
            f"Top prediction is {disease}. Triage level is {triage}. "
            + "Precautions are: "
            + ", ".join(precautions[:3])
        )
        audio_bytes = _tts_audio_bytes(speak_text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.warning("Could not generate voice output.")

    st.subheader("Chat Assistant")
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)

    st.caption("Voice chat: record your question and send it as a chat message.")
    voice_chat_audio = st.audio_input("Talk to assistant")
    send_voice = st.button("Send Voice Message", use_container_width=True)
    if send_voice:
        voice_chat_text = _transcribe_audio(voice_chat_audio)
        if voice_chat_text:
            st.info(f"You said: {voice_chat_text}")
            _handle_chat_message(
                voice_chat_text, disease, triage, precautions, red_flags
            )
        else:
            st.warning("Could not understand voice message. Try again clearly.")

    user_msg = st.chat_input("Ask the assistant (e.g., what should I do next?)")
    if user_msg:
        _handle_chat_message(user_msg, disease, triage, precautions, red_flags)

    if st.session_state.last_chat_reply_audio:
        st.audio(st.session_state.last_chat_reply_audio, format="audio/mp3")
