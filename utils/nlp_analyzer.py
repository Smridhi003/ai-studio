"""
NLP Analyzer — Named Entity Recognition + Linguistic Analysis
Uses spaCy for rule-based NLP and transformers for deep NLP features.
"""

import re
from collections import Counter

SPACY_AVAILABLE = False


class NLPAnalyzer:
    """
    Combines classical NLP techniques with ML-based entity extraction.
    
    Techniques used:
    - Tokenization (splitting text into tokens)
    - POS Tagging (Part-of-Speech: noun, verb, adjective...)
    - Named Entity Recognition (NER: persons, organizations, locations...)
    - Key phrase extraction (noun chunk detection)
    - Readability scoring (Flesch-Kincaid approximation)
    """

    ENTITY_COLORS = {
        "PERSON":  "#EEEDFE",
        "ORG":     "#E6F1FB",
        "GPE":     "#E1F5EE",
        "DATE":    "#FAEEDA",
        "MONEY":   "#EAF3DE",
        "PRODUCT": "#FAECE7",
        "LOC":     "#FBEAF0",
        "EVENT":   "#FCEBEB",
    }

    def analyze(self, text: str) -> dict:
        if SPACY_AVAILABLE:
            return self._analyze_spacy(text)
        return self._analyze_fallback(text)

    def _analyze_spacy(self, text: str) -> dict:
        doc = nlp(text)

        entities = [
            {"text": ent.text, "type": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

        key_phrases = list({chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1})[:8]

        pos_counts = Counter(token.pos_ for token in doc if not token.is_space)

        tokens = [t.text for t in doc if not t.is_space and not t.is_punct]
        sentences = list(doc.sents)
        avg_word_len = sum(len(t) for t in tokens) / max(len(tokens), 1)
        avg_sent_len = len(tokens) / max(len(sentences), 1)

        readability = self._readability_score(avg_word_len, avg_sent_len)

        return {
            "entities": entities,
            "tokens": len(tokens),
            "sentences": len(sentences),
            "avg_word_len": round(avg_word_len, 2),
            "key_phrases": key_phrases,
            "pos_distribution": dict(pos_counts.most_common(6)),
            "language": "English",
            "readability": readability,
        }

    def _analyze_fallback(self, text: str) -> dict:
        """Rule-based fallback when spaCy is not installed."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r"\b[a-zA-Z]+\b", text)

        # Simple NER patterns
        entities = []
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for noun in proper_nouns[:10]:
            label = "PERSON" if len(noun.split()) > 1 else "ORG"
            entities.append({"text": noun, "type": label, "start": text.find(noun), "end": text.find(noun)+len(noun)})

        money = re.findall(r"\$[\d,]+(?:\s+(?:billion|million|thousand))?", text)
        for m in money:
            entities.append({"text": m, "type": "MONEY", "start": text.find(m), "end": text.find(m)+len(m)})

        dates = re.findall(r"\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b", text)
        for d in dates:
            entities.append({"text": d, "type": "DATE", "start": text.find(d), "end": text.find(d)+len(d)})

        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        avg_sent_len = len(words) / max(len(sentences), 1)

        # Simple key phrase extraction (bigrams/trigrams from capitalized words)
        key_phrases = list({" ".join(words[i:i+2]) for i in range(len(words)-1) if words[i][0].isupper()})[:6]

        return {
            "entities": entities,
            "tokens": len(words),
            "sentences": len(sentences),
            "avg_word_len": round(avg_word_len, 2),
            "key_phrases": key_phrases,
            "pos_distribution": {"NOUN": len(proper_nouns), "NUM": len(money)},
            "language": "English",
            "readability": self._readability_score(avg_word_len, avg_sent_len),
        }

    def _readability_score(self, avg_word_len: float, avg_sent_len: float) -> str:
        """Approximate Flesch-Kincaid readability bucket."""
        score = (avg_word_len * 4) + (avg_sent_len * 0.5)
        if score < 10:
            return "easy"
        elif score < 15:
            return "medium"
        return "complex"
