# AI Voice Journal

A personal voice-first journal: record voice entries, transcribe, analyze sentiment, summarize, and visualize mood trends.

## Features
- Record audio (microphone) → transcribe to text
- Sentiment analysis (TextBlob or transformers)
- Summarization (transformers / fallback simple summarizer)
- Save entries to JSON
- Visualize mood over time with matplotlib

## Quick start (local)
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: `pyaudio` can be tricky to install on some systems. If you have issues, you can record audio externally and use the `transcribe_file` function.*
3. Run the CLI:
   ```bash
   python main.py
   ```

## Notes
- This repo uses optional AI libraries. The code has fallbacks so you can try the CLI without internet or heavy models.
- Replace `summarize_with_model` with your Groq or preferred LLM API call when ready.
