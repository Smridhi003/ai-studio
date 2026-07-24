"""
Text Generator — Controlled Autoregressive Language Model Generation
Uses GPT-2 (HuggingFace) for local generation, with a template-based
fallback for environments without GPU support.
"""

import random
from typing import Literal

TRANSFORMERS_AVAILABLE = False

LENGTH_TOKENS = {"short": 80, "medium": 200, "long": 400}

STYLE_PROMPTS = {
    "news":    "Breaking news: {topic} —",
    "tweet":   "1/ Thread on {topic}: ",
    "summary": "Executive Summary: Key developments in {topic}.",
    "eli5":    "Imagine you're 5 years old and someone asks about {topic}. Here's how to explain it simply:",
}

TEMPLATES = {
    "news": (
        "Scientists and researchers have made a significant breakthrough in the field of {topic}. "
        "According to reports, the latest developments are expected to reshape how we understand this domain. "
        "Experts from leading institutions have confirmed that the implications are far-reaching, "
        "potentially affecting millions of people worldwide. The findings were presented at a recent "
        "international conference and have already drawn significant attention from the global community. "
        "Further studies are underway to explore the full potential of this discovery."
    ),
    "tweet": (
        "1/ Big news on {topic} — here's what you need to know 🧵\n\n"
        "2/ The latest research shows major shifts in how we approach this. "
        "The implications are massive and we're just getting started.\n\n"
        "3/ Key takeaway: this changes the game for everyone involved. "
        "If you're not paying attention to {topic}, you're already behind.\n\n"
        "4/ Stay tuned for more updates. This thread will be updated as new info drops. "
        "#AI #Tech #{tag}"
    ),
    "summary": (
        "Executive Summary — {topic}\n\n"
        "Overview: This report provides a concise analysis of recent developments in {topic}. "
        "The landscape is evolving rapidly, driven by technological advancement and shifting market dynamics.\n\n"
        "Key Findings: Stakeholders have identified three critical areas of focus: innovation velocity, "
        "adoption barriers, and long-term sustainability. Data indicates a 40% increase in activity "
        "within the past 12 months.\n\n"
        "Recommendations: Organizations should prioritize investment in {topic} infrastructure, "
        "establish clear governance frameworks, and allocate dedicated resources for ongoing research."
    ),
    "eli5": (
        "Okay, so imagine {topic} is like a really cool toy that grown-ups use to solve big problems. "
        "You know how you use building blocks to make a tower? Well, scientists do something similar, "
        "but with invisible pieces called 'data' and 'algorithms'. "
        "Every time the toy sees something new, it gets a little smarter — just like how you learn "
        "new things every day at school! "
        "The more it practices, the better it gets. Pretty cool, right? "
        "That's basically what {topic} is all about!"
    ),
}


class TextGenerator:
    """
    Autoregressive text generation with style and length control.
    
    Demonstrates key DL concepts:
    - Conditional generation (prompt + style conditioning)
    - Temperature sampling vs. greedy decoding
    - Top-k and nucleus (top-p) sampling strategies
    - Length constraint via max_new_tokens
    """

    def generate(self, topic: str, style: str = "news", length: str = "medium") -> dict:
        max_tokens = LENGTH_TOKENS.get(length, 200)
        if TRANSFORMERS_AVAILABLE:
            return self._generate_gpt2(topic, style, max_tokens)
        return self._generate_template(topic, style)

    def _generate_gpt2(self, topic: str, style: str, max_tokens: int) -> dict:
        prompt = STYLE_PROMPTS.get(style, "{topic}:").format(topic=topic)
        set_seed(random.randint(0, 9999))
        outputs = _gen_pipe(
            prompt,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            temperature=0.85,
            top_p=0.92,
            top_k=50,
            do_sample=True,
            pad_token_id=50256,  # EOS token
        )
        generated = outputs[0]["generated_text"]
        return {
            "text":   generated,
            "style":  style,
            "length": max_tokens,
            "model":  "gpt2",
            "method": "nucleus-sampling (top-p=0.92, temperature=0.85)",
        }

    def _generate_template(self, topic: str, style: str) -> dict:
        template = TEMPLATES.get(style, TEMPLATES["news"])
        tag = topic.replace(" ", "").capitalize()
        text = template.format(topic=topic, tag=tag)
        return {
            "text":   text,
            "style":  style,
            "length": len(text.split()),
            "model":  "template-based",
            "method": "structured template with topic interpolation",
        }
