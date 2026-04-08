const API = "http://localhost:8000";

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.innerHTML = `<div class="bubble">${text}</div>`;
  messages.appendChild(div);
}

/* LOG SYMPTOM */
async function logSymptom() {
  const symptom = symptomInput.value;

  if (!symptom) return;

  // add to history
  const h = document.createElement("div");
  h.innerText = symptom;
  history.prepend(h);

  // call backend
  const res = await fetch(API + "/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ symptoms: [symptom] })
  });

  const data = await res.json();

  // update right panel
  precautions.innerHTML = "";
  data.Precautions.forEach(p => {
    precautions.innerHTML += `<li>${p}</li>`;
  });

  risk.innerHTML = `<li>${data.Triage}</li>`;

  if (data["Red Flags"]?.length) {
    alerts.innerHTML = "⚠ Emergency!";
  }

  // send to chat
  addMsg("user", "I have " + symptom);
  addMsg("ai", "Prediction: " + data["Predicted Disease"]);
}

/* CHAT */
async function sendMessage() {
  const text = chatInput.value;

  addMsg("user", text);

  const res = await fetch(API + "/claude", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      messages: [{ role: "user", content: text }],
      system_prompt: "health assistant"
    })
  });

  const data = await res.json();
  addMsg("ai", data.reply);
}

/* VOICE */
function startVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const rec = new SR();
  rec.start();

  rec.onresult = e => {
    chatInput.value = e.results[0][0].transcript;
    sendMessage();
  };
}
risk.innerHTML = "";

if(data.Triage === "high"){
  risk.innerHTML = `<div class="card risk-high">High Risk</div>`;
}
else if(data.Triage === "medium"){
  risk.innerHTML = `<div class="card risk-medium">Moderate Risk</div>`;
}
else{
  risk.innerHTML = `<div class="card risk-low">Low Risk</div>`;
}

/* ALERTS */
alerts.innerHTML = "";
if(data["Red Flags"]?.length){
  alerts.innerHTML += `<div class="alert warn">⚠ Emergency symptoms detected</div>`;
}