"""
Zero-Shot Text Classifier
Uses HuggingFace's zero-shot classification pipeline (NLI-based) or
a TF-IDF + cosine similarity fallback for lightweight environments.
"""

import re
import math
from typing import List

TRANSFORMERS_AVAILABLE = False


class ZeroShotClassifier:
    """
    Zero-shot classification using Natural Language Inference (NLI).
    
    The model is never trained on the target labels — it uses
    cross-lingual transfer from an MNLI (Multi-Genre NLI) pretrained
    model, treating each label as a hypothesis to entail or refute.
    
    Fallback: keyword overlap + TF-IDF cosine similarity.
    """

    def classify(self, text: str, labels: List[str]) -> dict:
        if TRANSFORMERS_AVAILABLE:
            return self._classify_transformers(text, labels)
        return self._classify_tfidf(text, labels)

    def _classify_transformers(self, text: str, labels: List[str]) -> dict:
        result = _clf_pipe(text[:512], candidate_labels=labels, multi_label=False)
        scores = dict(zip(result["labels"], result["scores"]))
        top = result["labels"][0]
        reasoning = f"The text most strongly entails the topic '{top}' based on semantic similarity and contextual cues."
        return {
            "scores":    {k: round(v, 4) for k, v in scores.items()},
            "top":       top,
            "reasoning": reasoning,
        }

    def _classify_tfidf(self, text: str, labels: List[str]) -> dict:
        """
        Lightweight fallback using TF-IDF cosine similarity.
        Expands each label with domain-relevant keywords.
        """
        label_keywords = {
            "Technology":     ["tech","software","hardware","digital","computer","ai","code","internet","device"],
            "Science":        ["research","study","scientist","experiment","discovery","biology","physics","lab"],
            "Business":       ["company","market","revenue","profit","CEO","stock","industry","trade","economy"],
            "Politics":       ["government","election","policy","president","law","senate","vote","party","bill"],
            "Sports":         ["game","team","player","match","score","win","league","coach","tournament"],
            "Entertainment":  ["movie","music","film","actor","show","celebrity","song","album","concert"],
            "Health":         ["medical","doctor","hospital","disease","treatment","vaccine","health","patient"],
            "Education":      ["school","student","teacher","university","degree","learning","curriculum","class"],
            "Environment":    ["climate","carbon","pollution","green","renewable","ecosystem","sustainability"],
            "Travel":         ["hotel","flight","destination","tourism","trip","vacation","travel","city"],
        }

        text_tokens = self._tokenize(text)
        text_tfidf  = self._tfidf_vector(text_tokens, [text])

        scores = {}
        for label in labels:
            # Build corpus from keywords if available
            keywords = label_keywords.get(label, label.lower().split())
            corpus_text = " ".join(keywords) + " " + label.lower()
            corpus_tokens = self._tokenize(corpus_text)
            label_tfidf = self._tfidf_vector(corpus_tokens, [corpus_text])
            scores[label] = self._cosine_similarity(text_tfidf, label_tfidf)

        total = sum(scores.values()) + 1e-9
        normalized = {k: round(v / total, 4) for k, v in scores.items()}
        top = max(normalized, key=normalized.get)

        return {
            "scores":    normalized,
            "top":       top,
            "reasoning": f"Based on keyword and semantic overlap, the text aligns most closely with '{top}'.",
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        stopwords = {"the","a","an","is","was","are","were","be","been","being",
                     "have","has","had","do","does","did","will","would","could",
                     "should","may","might","shall","can","of","to","in","for",
                     "on","with","at","by","from","as","into","through","during"}
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return [t for t in tokens if t not in stopwords]

    @staticmethod
    def _tfidf_vector(tokens: List[str], corpus: List[str]) -> dict:
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = max(len(tokens), 1)
        return {t: (c / total) for t, c in freq.items()}

    @staticmethod
    def _cosine_similarity(v1: dict, v2: dict) -> float:
        keys = set(v1) & set(v2)
        if not keys:
            return 0.0
        dot    = sum(v1[k] * v2[k] for k in keys)
        norm1  = math.sqrt(sum(v**2 for v in v1.values()))
        norm2  = math.sqrt(sum(v**2 for v in v2.values()))
        return dot / (norm1 * norm2 + 1e-9)
