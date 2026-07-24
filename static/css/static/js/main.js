/* Smridhi's AI/NLP Studio — main.js */

// ── Theme ──────────────────────────────────────────────────────────────────
const html = document.documentElement;
const themeBtn = document.getElementById("themeBtn");

function applyTheme(t) {
  html.setAttribute("data-theme", t);
  themeBtn.textContent = t === "dark" ? "☀️" : "🌙";
  localStorage.setItem("theme", t);
}
applyTheme(localStorage.getItem("theme") || "light");
themeBtn.addEventListener("click", () =>
  applyTheme(html.getAttribute("data-theme") === "dark" ? "light" : "dark")
);

// ── Tabs ───────────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("panel-" + tab.dataset.tab).classList.add("active");
  });
});

// ── Helpers ────────────────────────────────────────────────────────────────
const show = id => { const el = document.getElementById(id); el.classList.remove("hidden"); return el; };
const setLoading = (id, msg) => { show(id).innerHTML = `<span class="loading-text">⏳ ${msg}…</span>`; };
const setError   = (id, msg) => { show(id).innerHTML = `<span class="error-text">⚠️ ${msg}</span>`; };

async function post(endpoint, body) {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function bar(label, value, color, pct) {
  return `<div class="bar-row">
    <span class="bar-label">${label}</span>
    <div class="bar-wrap"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
    <span class="bar-val">${pct}%</span>
  </div>`;
}

function escapeHTML(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── NLP ────────────────────────────────────────────────────────────────────
async function runNLP() {
  const text = document.getElementById("nlp-input").value.trim();
  if (!text) return;
  setLoading("nlp-result", "Extracting entities");
  try {
    const d = await post("/api/nlp", { text });
    const cls = { PERSON:"PERSON",ORG:"ORG",GPE:"GPE",DATE:"DATE",MONEY:"MONEY",PRODUCT:"PRODUCT",LOC:"LOC" };
    let html = `<div class="entity-tags">`;
    (d.entities||[]).forEach(e => {
      const c = cls[e.type]||"default";
      html += `<span class="etag ${c}">${e.text}<span class="etag-type">${e.type}</span></span>`;
    });
    if (!d.entities||!d.entities.length) html += `<span style="color:var(--muted);font-size:13px">No named entities detected</span>`;
    html += `</div><hr class="sep">
    <div class="metric-grid">
      <div class="metric"><div class="metric-val">${d.tokens||0}</div><div class="metric-lbl">Tokens</div></div>
      <div class="metric"><div class="metric-val">${d.sentences||0}</div><div class="metric-lbl">Sentences</div></div>
      <div class="metric"><div class="metric-val">${d.avg_word_len||0}</div><div class="metric-lbl">Avg word len</div></div>
    </div>`;
    if (d.key_phrases&&d.key_phrases.length) {
      html += `<hr class="sep"><p class="sec-heading">Key phrases</p><div class="chips">`;
      d.key_phrases.forEach(p => { html += `<span class="chip">${p}</span>`; });
      html += `</div>`;
    }
    html += `<hr class="sep">
    <div class="info-row"><span class="info-key">Language</span><span class="info-value">${d.language||"English"}</span></div>
    <div class="info-row"><span class="info-key">Readability</span><span class="info-value">${d.readability||"—"}</span></div>`;
    document.getElementById("nlp-result").innerHTML = html;
  } catch(e) { setError("nlp-result","Analysis failed — is the server running?"); }
}

// ── Sentiment ──────────────────────────────────────────────────────────────
async function runSentiment() {
  const text = document.getElementById("sent-input").value.trim();
  if (!text) return;
  setLoading("sent-result","Analyzing sentiment");
  try {
    const d = await post("/api/sentiment", { text });
    const ov = d.overall||"neutral";
    const emColors = { joy:"#16a34a",anger:"#dc2626",fear:"#d97706",sadness:"#2563eb",surprise:"#7c3aed",disgust:"#0d9488" };
    let html = `<div class="sent-overview">
      <span class="sent-label">${ov.charAt(0).toUpperCase()+ov.slice(1)}</span>
      <span class="sent-badge ${ov}-badge">Score: ${(d.score||0).toFixed(2)}</span>
      <span style="font-size:12px;color:var(--muted)">Confidence: ${Math.round((d.confidence||0.8)*100)}%</span>
    </div><hr class="sep"><p class="sec-heading">Emotion breakdown</p>`;
    Object.entries(d.emotions||{}).forEach(([k,v]) => {
      html += bar(k.charAt(0).toUpperCase()+k.slice(1), v, emColors[k]||"#888", Math.round(v*100));
    });
    html += `<hr class="sep"><div class="info-row"><span class="info-key">Subjectivity</span><span class="info-value">${Math.round((d.subjectivity||0.5)*100)}%</span></div>`;
    if (d.aspects&&d.aspects.length) {
      html += `<hr class="sep"><p class="sec-heading">Aspect-level sentiment</p><div class="chips">`;
      d.aspects.forEach(a => {
        const c = a.sentiment==="positive"?"#16a34a":a.sentiment==="negative"?"#dc2626":"#2563eb";
        html += `<span class="chip" style="background:${c}18;color:${c};border-color:${c}44">${a.aspect} · ${a.sentiment}</span>`;
      });
      html += `</div>`;
    }
    document.getElementById("sent-result").innerHTML = html;
  } catch(e) { setError("sent-result","Sentiment analysis failed."); }
}

// ── Classify ───────────────────────────────────────────────────────────────
async function runClassify() {
  const text   = document.getElementById("clf-input").value.trim();
  const labStr = document.getElementById("clf-labels").value.trim();
  if (!text||!labStr) return;
  const labels = labStr.split(",").map(l=>l.trim()).filter(Boolean);
  setLoading("clf-result","Classifying");
  try {
    const d = await post("/api/classify", { text, labels });
    const colors = ["#4f46e5","#0891b2","#16a34a","#d97706","#dc2626","#7c3aed","#db2777","#0d9488"];
    let html = `<p class="sec-heading">Classification probabilities</p>`;
    Object.entries(d.scores||{}).sort((a,b)=>b[1]-a[1]).forEach(([label,score],i) => {
      const isTop = label===d.top;
      html += `<div class="bar-row">
        <span class="bar-label" style="${isTop?"font-weight:600;color:var(--text)":""}">${label}</span>
        <div class="bar-wrap"><div class="bar-fill" style="width:${Math.round(score*100)}%;background:${colors[i%colors.length]}${isTop?"":"88"}"></div></div>
        <span class="bar-val">${Math.round(score*100)}%</span>
      </div>`;
    });
    if (d.reasoning) html += `<hr class="sep"><p style="font-size:13px;color:var(--muted);line-height:1.7">${d.reasoning}</p>`;
    document.getElementById("clf-result").innerHTML = html;
  } catch(e) { setError("clf-result","Classification failed."); }
}

// ── Generate ───────────────────────────────────────────────────────────────
async function runGenerate() {
  const topic  = document.getElementById("gen-input").value.trim();
  const style  = document.getElementById("gen-style").value;
  const length = document.getElementById("gen-len").value;
  if (!topic) return;
  setLoading("gen-result","Generating text");
  try {
    const d = await post("/api/generate", { topic, style, length });
    const html = `<div class="gen-meta">
      <span>Style: ${d.style}</span><span>Model: ${d.model}</span><span>Method: ${d.method}</span>
    </div><div class="gen-text">${escapeHTML(d.text)}</div>`;
    document.getElementById("gen-result").innerHTML = html;
  } catch(e) { setError("gen-result","Generation failed."); }
}

// ── Neural Net ─────────────────────────────────────────────────────────────
async function runNeuralNet() {
  const text = document.getElementById("nn-input").value.trim();
  if (!text) return;
  setLoading("nn-result","Running forward pass");
  try {
    const d = await post("/api/neural-net", { text });
    const lColors = ["#4f46e5","#0891b2","#16a34a","#d97706"];
    const lBg     = ["#eef2ff","#ecfeff","#f0fdf4","#fffbeb"];
    const lNames  = ["Input","Hidden 1\n(ReLU)","Hidden 2\n(Sigmoid)","Output\n(Softmax)"];
    let diagram = `<div class="nn-diagram">`;
    (d.layer_activations||[]).forEach((neurons,li) => {
      if (li>0) {
        diagram += `<div class="nn-connector">`;
        for(let i=0;i<neurons.length;i++) diagram += `<div class="conn-line"></div>`;
        diagram += `</div>`;
      }
      diagram += `<div class="nn-layer">`;
      neurons.forEach(v => {
        diagram += `<div class="neuron" style="border-color:${lColors[li]};background:${lBg[li]};opacity:${Math.max(0.18,v).toFixed(2)};color:${lColors[li]}">${v.toFixed(1)}</div>`;
      });
      diagram += `<div class="nn-layer-label">${lNames[li]||""}</div></div>`;
    });
    diagram += `</div>`;

    const pColors = { positive:"#16a34a",negative:"#dc2626",neutral:"#2563eb" };
    let probs = `<hr class="sep"><p class="sec-heading">Output probabilities</p>`;
    Object.entries(d.probabilities||{}).sort((a,b)=>b[1]-a[1]).forEach(([k,v]) => {
      probs += bar(k.charAt(0).toUpperCase()+k.slice(1), v, pColors[k]||"#888", Math.round(v*100));
    });
    probs += `<hr class="sep">
    <div class="info-row"><span class="info-key">Prediction</span><span class="info-value">${d.prediction||"—"}</span></div>
    <div class="info-row"><span class="info-key">Confidence</span><span class="info-value">${Math.round((d.confidence||0)*100)}%</span></div>`;
    if (d.architecture) {
      probs += `<hr class="sep"><p class="sec-heading">Architecture</p>`;
      d.architecture.layers.forEach((l,i) => {
        probs += `<div class="info-row"><span class="info-key">Layer ${i}</span><span class="info-value">${l}</span></div>`;
      });
      probs += `<div class="info-row"><span class="info-key">Parameters</span><span class="info-value">${d.architecture.params}</span></div>`;
    }
    if (d.input_features) {
      probs += `<hr class="sep"><p class="sec-heading">Input feature vector</p>`;
      Object.entries(d.input_features).forEach(([k,v]) => {
        probs += bar(k.replace(/_/g," ").replace(/^\w/,c=>c.toUpperCase()), v, "#6b7280", Math.round(v*100));
      });
    }
    document.getElementById("nn-result").innerHTML = diagram + probs;
  } catch(e) { setError("nn-result","Forward pass failed."); }
}

// ── Chat ───────────────────────────────────────────────────────────────────
let chatHistory = [];

function quickChat(msg) {
  document.getElementById("chatInput").value = msg;
  sendChat();
}

async function sendChat() {
  const input = document.getElementById("chatInput");
  const msg   = input.value.trim();
  if (!msg) return;
  input.value = "";

  const win = document.getElementById("chatWindow");

  // User bubble
  win.innerHTML += `<div class="chat-msg user">
    <div class="chat-avatar user-av">U</div>
    <div class="chat-bubble">${escapeHTML(msg)}</div>
  </div>`;

  // Typing indicator
  const typingId = "typing-" + Date.now();
  win.innerHTML += `<div class="chat-msg bot" id="${typingId}">
    <div class="chat-avatar bot-av">S</div>
    <div class="chat-bubble chat-typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
  </div>`;
  win.scrollTop = win.scrollHeight;

  chatHistory.push({ role:"user", content: msg });

  try {
    const d = await post("/api/chat", { message: msg, history: chatHistory });
    chatHistory.push({ role:"assistant", content: d.reply });
    document.getElementById(typingId).outerHTML = `<div class="chat-msg bot">
      <div class="chat-avatar bot-av">S</div>
      <div class="chat-bubble">${escapeHTML(d.reply)}</div>
    </div>`;
  } catch(e) {
    document.getElementById(typingId).outerHTML = `<div class="chat-msg bot">
      <div class="chat-avatar bot-av">S</div>
      <div class="chat-bubble" style="color:#ef4444">Sorry, something went wrong. Is the server running?</div>
    </div>`;
  }
  win.scrollTop = win.scrollHeight;
}
