# ⬡ Smridhi · AI/NLP Studio

> **CSE-IoT Portfolio Project** — A unified platform combining Natural Language Processing, Machine Learning, Deep Learning, and AI Chat — built with Flask, HuggingFace Transformers, spaCy, and NumPy.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![spaCy](https://img.shields.io/badge/spaCy-3.7-09a3d5)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Tab | Technique | What it does |
|---|---|---|
| 🔤 **NLP Analysis** | NER + POS + Tokenization | Entities, key phrases, readability |
| 😊 **Sentiment** | RoBERTa Transformer | 6 emotions + aspect-level sentiment |
| 🏷️ **Classify** | Zero-Shot NLI (BART) | Any labels, no training needed |
| ✨ **Generate** | GPT-2 autoregressive | News, tweet, summary, ELI5 styles |
| 🧠 **Neural Net** | Real DNN in NumPy | Live forward pass visualization |
| 💬 **AI Chat** | Rule-based + T5 | Ask anything about the project |
| 👩‍💻 **About** | Portfolio section | Smridhi's skills and contact |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/smridhi-ai-studio.git
cd smridhi-ai-studio

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install (lightweight, no GPU needed)
pip install -r requirements-lite.txt

# 4. Run
python app.py
# Open http://localhost:5000
```

For full HuggingFace models (better accuracy):
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
```

---

## 🗂️ Project Structure

```
smridhi-ai-studio/
├── app.py                    # Flask app + 6 API routes
├── utils/
│   ├── nlp_analyzer.py       # NER, POS, tokenization
│   ├── sentiment.py          # Emotion + sentiment detection
│   ├── classifier.py         # Zero-shot classification
│   ├── generator.py          # Text generation
│   ├── neural_net.py         # DNN forward pass (NumPy)
│   └── chatbot.py            # AI chat assistant
├── templates/index.html      # Full UI (dark mode + chat + about)
├── static/
│   ├── css/style.css         # Light/dark theme styles
│   └── js/main.js            # Frontend logic
├── notebooks/
│   └── exploration.ipynb     # Jupyter experiments
├── requirements.txt          # Full (HuggingFace)
├── requirements-lite.txt     # Lightweight (no GPU)
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Body |
|---|---|---|
| POST | `/api/nlp` | `{ "text": "..." }` |
| POST | `/api/sentiment` | `{ "text": "..." }` |
| POST | `/api/classify` | `{ "text": "...", "labels": [...] }` |
| POST | `/api/generate` | `{ "topic": "...", "style": "news", "length": "medium" }` |
| POST | `/api/neural-net` | `{ "text": "..." }` |
| POST | `/api/chat` | `{ "message": "...", "history": [...] }` |
| GET  | `/health` | — |

---

## 🧠 Techniques Covered

- **NLP:** Tokenization, POS Tagging, Named Entity Recognition, Readability scoring
- **ML:** Zero-shot NLI classification, TF-IDF cosine similarity, Lexicon-based sentiment
- **DL:** Transformer models (RoBERTa, BART, GPT-2), Feedforward DNN with NumPy
- **Generation:** Nucleus sampling (top-p), Temperature scaling, Style conditioning
- **IoT AI:** Designed for edge/sensor integration — NLP on device logs, IoT data streams

---

## 🌍 Deploy (Free)

**Render.com or Railway.app:**
```
Start command: gunicorn app:app --bind 0.0.0.0:$PORT
```

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-lite.txt .
RUN pip install -r requirements-lite.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

---

## 👩‍💻 Author

**Smridhi** — CSE-IoT Student | AI/ML Enthusiast

- 🔗 [GitHub](https://github.com/Smridhi003)
- 💼 [LinkedIn](https://www.linkedin.com/in/smridhi-ba7ba3373?utm_source=share_via&utm_content=profile&utm_medium=member_android)

---

## 📄 License

MIT — free to use, learn from, and build on.
