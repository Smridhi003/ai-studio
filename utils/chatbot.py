"""
Chatbot — Smridhi's AI Chat Assistant
Context-aware conversational AI using a lightweight rule-based + retrieval
approach, with an optional HuggingFace pipeline for richer responses.

The bot knows about:
- Smridhi's project (what each module does)
- NLP, ML, DL, IoT concepts
- General AI Q&A
"""

import re
import random
from typing import List, Dict

TRANSFORMERS_AVAILABLE = False


# ── Knowledge base ─────────────────────────────────────────────────────────
KB = {
    r"(who (are|is) (you|this)|what (is|are) (you|this app)|about this)":
        "I'm Smridhi's AI Studio — a portfolio project built with Flask, spaCy, HuggingFace Transformers, and NumPy. "
        "It combines NLP, ML, DL, and IoT-aware AI in one platform. Ask me about any of the modules!",

    r"(who (made|built|created) (you|this)|smridhi)":
        "This project was built by Smridhi, a CSE-IoT student passionate about AI and intelligent systems. "
        "The studio showcases real ML/DL/NLP techniques applied together.",

    r"(nlp|natural language processing|named entity|ner|pos tag)":
        "The NLP module uses spaCy for Named Entity Recognition (NER), POS tagging, and tokenization. "
        "NER identifies PERSON, ORG, DATE, MONEY, and GPE entities. POS tagging labels every word as noun, verb, adjective etc. "
        "There's also a rule-based fallback that works without any installed models.",

    r"(sentiment|emotion|feeling|mood|positive|negative)":
        "The Sentiment module runs a RoBERTa transformer to detect 6 emotions: joy, anger, fear, sadness, surprise, disgust. "
        "It goes beyond simple positive/negative — it also computes subjectivity and aspect-level sentiment per clause.",

    r"(classif|zero.?shot|label|categor)":
        "The Zero-Shot Classifier uses BART-MNLI (Natural Language Inference) to classify text into any labels you define — "
        "without any task-specific training data. This is a powerful transfer learning technique.",

    r"(generat|gpt|language model|autoregress|text generat)":
        "The Generation module uses GPT-2 with nucleus sampling (top-p=0.92, temp=0.85) for diverse, coherent output. "
        "You can pick style (news, tweet, summary, ELI5) and length. The method conditions generation on your topic.",

    r"(neural|deep learning|forward pass|neuron|layer|relu|sigmoid|softmax)":
        "The Neural Net visualizer runs a real 5→5→5→3 feedforward DNN implemented in NumPy. "
        "It extracts 5 text features (positive density, negative density, intensifier ratio, question ratio, exclamation ratio), "
        "passes them through ReLU and Sigmoid hidden layers, then Softmax output. You can see live activation values per neuron.",

    r"(iot|internet of things|embedded|sensor|device|edge)":
        "IoT + AI is Smridhi's core focus area. Deploying ML on edge devices, processing sensor streams with NLP, "
        "and building intelligent IoT dashboards are natural extensions of this studio. "
        "Think sentiment analysis on user feedback from IoT apps, or NER on device logs.",

    r"(dark mode|theme|light|appearance)":
        "You can toggle dark/light mode using the moon/sun button in the top-right corner. "
        "Your preference is saved to localStorage so it persists across sessions.",

    r"(how (do i|to) use|help|guide|tutorial)":
        "Easy! Pick a tab at the top: NLP Analysis, Sentiment, Classify, Generate, Neural Net, or this Chat. "
        "Type your text, hit the run button, and see results instantly. Each module is independent.",

    r"(install|setup|run|start|flask|python|requirement)":
        "To run locally: (1) pip install -r requirements-lite.txt (2) python app.py (3) open http://localhost:5000. "
        "For full HuggingFace models use requirements.txt instead. Models download automatically on first run.",

    r"(github|portfolio|resume|job|placement|hire)":
        "This project is designed as a portfolio piece for Smridhi. It demonstrates full-stack AI development: "
        "backend ML APIs (Flask), multiple NLP/DL techniques, clean code structure, and a polished UI. "
        "Push it with 'git push origin main' and add the live URL to your resume!",

    r"(transformers|huggingface|bert|roberta|bart|gpt)":
        "HuggingFace Transformers power three modules: RoBERTa for sentiment, BART-MNLI for zero-shot classification, "
        "and GPT-2 for generation. All have fallbacks so the app runs without GPU too.",

    r"(spacy|en_core|model download)":
        "spaCy powers the NLP module. After installing, run: python -m spacy download en_core_web_sm "
        "This downloads the small English model (~12MB) for NER and POS tagging.",

    r"(deploy|heroku|render|railway|production|live)":
        "To deploy: use 'gunicorn app:app' (already in requirements). "
        "Render.com and Railway.app both offer free tiers that work great with Flask. "
        "Set your start command to: gunicorn app:app --bind 0.0.0.0:$PORT",
}

GREETINGS = ["hi", "hello", "hey", "hii", "helo", "yo", "sup", "namaste"]
FAREWELLS  = ["bye", "goodbye", "see you", "thanks", "thank you", "cya"]

GREETING_REPLIES = [
    "Hey! 👋 I'm Smridhi's AI assistant. Ask me about any module — NLP, Sentiment, Classification, Generation, or the Neural Net!",
    "Hello! I'm the built-in AI for this studio. What would you like to know?",
    "Hi there! Ask me anything about this project, the AI techniques used, or how to run it.",
]
FAREWELL_REPLIES = [
    "Thanks for visiting Smridhi's AI Studio! Good luck! 🚀",
    "Bye! Feel free to come back and explore more modules.",
    "See you! Hope the project was helpful for your learning.",
]

FALLBACKS = [
    "Hmm, I'm not sure about that one. Try asking about a specific module — NLP, Sentiment, Classification, Generation, or the Neural Net!",
    "I don't have a specific answer for that, but you can explore the tabs above to see each AI module in action.",
    "That's outside my knowledge base right now. Try rephrasing, or ask me about how this project works!",
]


class Chatbot:
    """
    Rule-based + retrieval chatbot with optional T5 fallback.
    Maintains context via conversation history.
    """

    def respond(self, message: str, history: List[Dict] = None) -> dict:
        msg_lower = message.lower().strip()

        # Greetings
        if any(g in msg_lower.split() or msg_lower == g for g in GREETINGS):
            return {"reply": random.choice(GREETING_REPLIES), "source": "rule"}

        # Farewells
        if any(f in msg_lower for f in FAREWELLS):
            return {"reply": random.choice(FAREWELL_REPLIES), "source": "rule"}

        # Knowledge base lookup
        for pattern, answer in KB.items():
            if re.search(pattern, msg_lower, re.IGNORECASE):
                return {"reply": answer, "source": "kb"}

        # Optional T5 generation for open questions
        if TRANSFORMERS_AVAILABLE:
            try:
                prompt = f"Answer this question about an AI/NLP project: {message}"
                out = _qa_pipe(prompt, max_new_tokens=120, do_sample=False)
                reply = out[0]["generated_text"].strip()
                if reply and len(reply) > 20:
                    return {"reply": reply, "source": "t5"}
            except Exception:
                pass

        return {"reply": random.choice(FALLBACKS), "source": "fallback"}
