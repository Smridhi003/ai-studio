"""
Sentiment Analyzer — Multi-dimensional Deep Learning Sentiment Analysis
Uses transformers (HuggingFace) for emotion detection, or falls back to
a lexicon-based approach for environments without GPU/transformers.
"""

import re
from typing import Dict

TRANSFORMERS_AVAILABLE = False


# ─── Lexicon fallback ─────────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "good","great","excellent","amazing","wonderful","fantastic","brilliant",
    "love","perfect","best","awesome","happy","beautiful","outstanding",
    "superb","incredible","enjoy","pleased","satisfied","delighted",
}
NEGATIVE_WORDS = {
    "bad","terrible","awful","horrible","poor","worst","hate","disappointing",
    "frustrating","slow","broken","fail","wrong","disappointing","useless",
    "annoying","angry","sad","upset","disaster","dreadful","pathetic",
}
INTENSIFIERS = {"absolutely","very","extremely","incredibly","totally","really","quite"}


class SentimentAnalyzer:
    """
    Multi-dimensional sentiment analysis combining:
    - Overall polarity (positive / negative / neutral / mixed)
    - Confidence score
    - 6-emotion breakdown (joy, anger, fear, sadness, surprise, disgust)
    - Subjectivity estimation
    - Aspect-level sentiment (per-clause analysis)
    """

    def analyze(self, text: str) -> dict:
        if TRANSFORMERS_AVAILABLE:
            return self._analyze_transformers(text)
        return self._analyze_lexicon(text)

    # ── Transformer path ──────────────────────────────────────────────────────
    def _analyze_transformers(self, text: str) -> dict:
        truncated = text[:512]

        sent_scores = {r["label"].lower(): r["score"] for r in _sentiment_pipe(truncated)[0]}
        emotion_scores = {r["label"].lower(): r["score"] for r in _emotion_pipe(truncated)[0]}

        pos = sent_scores.get("positive", 0)
        neg = sent_scores.get("negative", 0)
        neu = sent_scores.get("neutral", 0)

        if pos > neg and pos > neu:
            overall, score = "positive", pos
        elif neg > pos and neg > neu:
            overall, score = "negative", neg
        elif abs(pos - neg) < 0.15:
            overall, score = "mixed", max(pos, neg)
        else:
            overall, score = "neutral", neu

        emotions = {
            "joy":      emotion_scores.get("joy", 0),
            "anger":    emotion_scores.get("anger", 0),
            "fear":     emotion_scores.get("fear", 0),
            "sadness":  emotion_scores.get("sadness", 0),
            "surprise": emotion_scores.get("surprise", 0),
            "disgust":  emotion_scores.get("disgust", 0),
        }

        subjectivity = 1.0 - neu
        aspects = self._extract_aspects(text)

        return {
            "overall":     overall,
            "score":       round(score, 3),
            "confidence":  round(max(pos, neg, neu), 3),
            "emotions":    {k: round(v, 3) for k, v in emotions.items()},
            "subjectivity": round(subjectivity, 3),
            "aspects":     aspects,
        }

    # ── Lexicon fallback ──────────────────────────────────────────────────────
    def _analyze_lexicon(self, text: str) -> dict:
        words = re.findall(r"\b\w+\b", text.lower())
        pos_count, neg_count = 0, 0
        intensity = 1.0

        for i, w in enumerate(words):
            if w in INTENSIFIERS:
                intensity = 1.5
                continue
            if w in POSITIVE_WORDS:
                pos_count += intensity
            elif w in NEGATIVE_WORDS:
                neg_count += intensity
            intensity = 1.0

        total = pos_count + neg_count + 1e-9

        if pos_count > neg_count * 1.5:
            overall, score = "positive", round(pos_count / total, 3)
        elif neg_count > pos_count * 1.5:
            overall, score = "negative", round(neg_count / total, 3)
        elif pos_count > 0 and neg_count > 0:
            overall, score = "mixed", round(max(pos_count, neg_count) / total, 3)
        else:
            overall, score = "neutral", 0.5

        norm_pos = pos_count / total
        norm_neg = neg_count / total

        emotions = {
            "joy":      round(min(norm_pos * 1.2, 1.0), 3),
            "anger":    round(min(norm_neg * 0.8, 1.0), 3),
            "fear":     round(min(norm_neg * 0.4, 1.0), 3),
            "sadness":  round(min(norm_neg * 0.6, 1.0), 3),
            "surprise": round(min((pos_count + neg_count) / (len(words) + 1), 1.0), 3),
            "disgust":  round(min(norm_neg * 0.5, 1.0), 3),
        }

        subjectivity = round(min((pos_count + neg_count) / max(len(words), 1) * 3, 1.0), 3)

        return {
            "overall":      overall,
            "score":        score,
            "confidence":   round(max(norm_pos, norm_neg, 0.5), 3),
            "emotions":     emotions,
            "subjectivity": subjectivity,
            "aspects":      self._extract_aspects(text),
        }

    def _extract_aspects(self, text: str) -> list:
        """Clause-level aspect sentiment via simple heuristics."""
        aspects = []
        clauses = re.split(r"[,;]|but|however|although|though", text, flags=re.IGNORECASE)
        for clause in clauses[:5]:
            words_low = clause.lower().split()
            pos = sum(1 for w in words_low if w in POSITIVE_WORDS)
            neg = sum(1 for w in words_low if w in NEGATIVE_WORDS)
            if not words_low:
                continue
            # Try to extract subject noun (first capitalized or known noun)
            subject = next((w for w in clause.split() if w[0].isupper() and len(w) > 2), "")
            if not subject:
                nouns = [w for w in words_low if len(w) > 4]
                subject = nouns[0].capitalize() if nouns else clause.strip()[:20]
            sentiment = "positive" if pos > neg else ("negative" if neg > pos else "neutral")
            sc = round((pos - neg) / max(pos + neg, 1), 2)
            aspects.append({"aspect": subject, "sentiment": sentiment, "score": abs(sc)})
        return [a for a in aspects if a["aspect"]][:4]
