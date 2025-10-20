"""
summarizer.py
Provides a basic summarizer. Try transformers summarizer if available, otherwise fallback to simple sentence selection.
"""

import re

try:
    from transformers import pipeline
    _hf_summarizer = pipeline("summarization")
except Exception:
    _hf_summarizer = None

def summarize_text(text, max_length=130):
    if _hf_summarizer is not None:
        try:
            out = _hf_summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return out[0]["summary_text"]
        except Exception:
            pass
    # Fallback: extract top 3 longest sentences as a naive "summary"
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences.sort(key=lambda s: len(s), reverse=True)
    return " ".join(sentences[:3])
