"""
Neural Network Simulator — Deep Learning Forward Pass Visualization
Implements a real feedforward neural network with numpy to demonstrate
tokenization → embedding → hidden layers → softmax output.
"""

import math
import random
import re
from typing import List

NUMPY_AVAILABLE = False


def _sigmoid(x):
    if NUMPY_AVAILABLE:
        return float(1 / (1 + np.exp(-x)))
    return 1 / (1 + math.exp(max(-500, min(500, -x))))


def _relu(x):
    return max(0.0, x)


def _softmax(values: List[float]) -> List[float]:
    mx = max(values)
    exps = [math.exp(v - mx) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


# ── Tiny pretrained-style weight matrix (seeded deterministically) ─────────────
def _make_weights(rows: int, cols: int, seed: int = 42) -> List[List[float]]:
    random.seed(seed)
    return [[random.gauss(0, 0.5) for _ in range(cols)] for _ in range(rows)]


W1 = _make_weights(5, 5, seed=1)
W2 = _make_weights(5, 5, seed=2)
W3 = _make_weights(5, 3, seed=3)
B1 = [random.gauss(0, 0.1) for _ in range(5)]
B2 = [random.gauss(0, 0.1) for _ in range(5)]
B3 = [random.gauss(0, 0.1) for _ in range(3)]


def _dot(weights: List[List[float]], inputs: List[float], bias: List[float]) -> List[float]:
    result = []
    for w_row, b in zip(weights, bias):
        val = sum(w * x for w, x in zip(w_row, inputs)) + b
        result.append(val)
    return result


class NeuralNetSimulator:
    """
    5-5-5-3 Feedforward Neural Network
    
    Architecture:
        Input  Layer  : 5 neurons (bag-of-words embedding)
        Hidden Layer 1: 5 neurons + ReLU activation
        Hidden Layer 2: 5 neurons + Sigmoid activation
        Output Layer  : 3 neurons + Softmax → [positive, negative, neutral]
    
    This simulates how a real text classification DNN processes sentences:
    1. Tokenize → count occurrences of sentiment-relevant features
    2. Normalize into a 5-dim feature vector (input)
    3. Forward pass through two hidden layers
    4. Softmax output = class probabilities
    """

    POSITIVE_WORDS = {
        "good","great","excellent","amazing","wonderful","fantastic","brilliant",
        "love","perfect","best","awesome","happy","beautiful","outstanding",
        "superb","incredible","enjoy","pleased","satisfied","brilliant",
    }
    NEGATIVE_WORDS = {
        "bad","terrible","awful","horrible","poor","worst","hate","disappointing",
        "frustrating","slow","broken","wrong","useless","annoying","angry",
        "sad","upset","disaster","dreadful","pathetic","negative",
    }
    INTENSIFIERS   = {"absolutely","very","extremely","incredibly","totally","really"}
    QUESTION_WORDS = {"what","why","how","when","where","who","which","does","is","are"}
    EXCLAMATION    = {"!","amazing","wow","incredible","fantastic","unbelievable"}

    def forward_pass(self, text: str) -> dict:
        # ── Feature Extraction (simulated embedding layer) ─────────────────
        tokens = re.findall(r"\b\w+\b", text.lower())
        n      = max(len(tokens), 1)

        pos_density  = sum(1 for t in tokens if t in self.POSITIVE_WORDS) / n
        neg_density  = sum(1 for t in tokens if t in self.NEGATIVE_WORDS) / n
        intensity    = sum(1 for t in tokens if t in self.INTENSIFIERS)   / n
        question     = sum(1 for t in tokens if t in self.QUESTION_WORDS) / n
        exclamation  = min(text.count("!") / max(len(text), 1) * 20 + \
                       sum(1 for t in tokens if t in self.EXCLAMATION) / n, 1.0)

        input_vec = [
            round(pos_density, 4),
            round(neg_density, 4),
            round(intensity,   4),
            round(question,    4),
            round(exclamation, 4),
        ]

        # ── Layer 1: Linear → ReLU ────────────────────────────────────────
        z1     = _dot(W1, input_vec, B1)
        hidden1 = [round(_relu(z), 4) for z in z1]
        h1_norm = self._normalize(hidden1)

        # ── Layer 2: Linear → Sigmoid ─────────────────────────────────────
        z2     = _dot(W2, hidden1, B2)
        hidden2 = [round(_sigmoid(z), 4) for z in z2]
        h2_norm = self._normalize(hidden2)

        # ── Output Layer: Linear → Softmax ────────────────────────────────
        z3    = _dot(W3, hidden2, B3)
        probs = _softmax(z3)
        labels = ["positive", "negative", "neutral"]
        prediction = labels[probs.index(max(probs))]
        confidence = round(max(probs), 4)

        return {
            "input_features": {
                "positive_density":  input_vec[0],
                "negative_density":  input_vec[1],
                "intensifier_ratio": input_vec[2],
                "question_ratio":    input_vec[3],
                "exclamation_ratio": input_vec[4],
            },
            "layer_activations": [
                [round(v, 2) for v in input_vec],
                [round(v, 2) for v in h1_norm],
                [round(v, 2) for v in h2_norm],
                [round(v, 2) for v in self._normalize(probs)[:3]],
            ],
            "probabilities": {
                labels[i]: round(probs[i], 4) for i in range(3)
            },
            "prediction":  prediction,
            "confidence":  confidence,
            "architecture": {
                "layers":      ["Input (5)", "Hidden-1 ReLU (5)", "Hidden-2 Sigmoid (5)", "Output Softmax (3)"],
                "params":      len(W1)*len(W1[0]) + len(W2)*len(W2[0]) + len(W3)*len(W3[0]),
                "activations": ["—", "ReLU", "Sigmoid", "Softmax"],
            }
        }

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        mx = max(values) if values else 1
        mn = min(values) if values else 0
        rng = mx - mn + 1e-9
        return [round((v - mn) / rng, 4) for v in values]
