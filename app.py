"""
Smridhi's AI/ML/DL/NLP Studio
CSE-IoT | Portfolio Project

A unified platform combining NLP, Sentiment Analysis, Text Classification,
Text Generation, Neural Network Visualization, and an AI Chat Assistant.
"""

from flask import Flask, render_template, request, jsonify
from utils.nlp_analyzer import NLPAnalyzer
from utils.sentiment import SentimentAnalyzer
from utils.classifier import ZeroShotClassifier
from utils.generator import TextGenerator
from utils.neural_net import NeuralNetSimulator
from utils.chatbot import Chatbot
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

nlp_analyzer      = NLPAnalyzer()
sentiment_analyzer = SentimentAnalyzer()
classifier        = ZeroShotClassifier()
generator         = TextGenerator()
neural_net        = NeuralNetSimulator()
chatbot           = Chatbot()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/nlp", methods=["POST"])
def analyze_nlp():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        return jsonify(nlp_analyzer.analyze(text))
    except Exception as e:
        logger.error(f"NLP error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sentiment", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        return jsonify(sentiment_analyzer.analyze(text))
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/classify", methods=["POST"])
def classify_text():
    data = request.get_json()
    text   = data.get("text", "").strip()
    labels = data.get("labels", [])
    if not text or not labels:
        return jsonify({"error": "Text and labels required"}), 400
    try:
        return jsonify(classifier.classify(text, labels))
    except Exception as e:
        logger.error(f"Classify error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate_text():
    data   = request.get_json()
    topic  = data.get("topic", "").strip()
    style  = data.get("style", "news")
    length = data.get("length", "medium")
    if not topic:
        return jsonify({"error": "Topic required"}), 400
    try:
        return jsonify(generator.generate(topic, style, length))
    except Exception as e:
        logger.error(f"Generate error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/neural-net", methods=["POST"])
def run_neural_net():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        return jsonify(neural_net.forward_pass(text))
    except Exception as e:
        logger.error(f"Neural net error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = data.get("message", "").strip()
    history = data.get("history", [])
    if not message:
        return jsonify({"error": "No message provided"}), 400
    try:
        return jsonify(chatbot.respond(message, history))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "author": "Smridhi",
        "branch": "CSE-IoT",
        "modules": ["nlp", "sentiment", "classify", "generate", "neural-net", "chat"]
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
