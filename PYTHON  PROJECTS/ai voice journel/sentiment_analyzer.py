"""
sentiment_analyzer.py
Provides a simple sentiment analysis wrapper. Uses TextBlob by default with a transformers fallback.
"""

from textblob import TextBlob

# Try transformers-based sentiment if available
try:
    from transformers import pipeline
    _hf_sentiment = pipeline("sentiment-analysis")
except Exception:
    _hf_sentiment = None

def analyze_sentiment(text):
    """
    Returns a dict like: {"label":"POSITIVE","score":0.95, "polarity": 0.3}
    Polarity from TextBlob for numeric trend plotting.
    """
    tb = TextBlob(text)
    polarity = tb.sentiment.polarity  # -1 to 1
    result = {"polarity": polarity, "label": "NEUTRAL", "score": abs(polarity)}
    if _hf_sentiment is not None:
        try:
            hf = _hf_sentiment(text[:512])[0]
            result.update({"label": hf.get("label"), "score": float(hf.get("score", 0))})
        except Exception:
            pass
    else:
        # Derive coarse label from polarity
        if polarity > 0.2:
            result["label"] = "POSITIVE"
        elif polarity < -0.2:
            result["label"] = "NEGATIVE"
        else:
            result["label"] = "NEUTRAL"
    return result
