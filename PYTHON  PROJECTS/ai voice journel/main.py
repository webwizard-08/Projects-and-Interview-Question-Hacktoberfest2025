"""
main.py
Simple CLI to record or transcribe, analyze, summarize, save, and view entries.
"""

import sys
from pathlib import Path

from sentiment_analyzer import analyze_sentiment
from summarizer import summarize_text
from data_logger import save_entry, load_entries
from visualizer import plot_sentiment

# Lazy import for audio functions (optional)
try:
    from voice_input import record_to_file, transcribe_file
except Exception:
    record_to_file = None
    transcribe_file = None

def create_entry_from_text(text):
    sentiment = analyze_sentiment(text)
    summary = summarize_text(text)
    entry = save_entry(text, sentiment, summary)
    print("Saved entry:")
    print("Timestamp:", entry["timestamp"])
    print("Summary:", entry["summary"])
    print("Sentiment:", entry["sentiment"])
    return entry

def cli_menu():
    while True:
        print("\nAI Voice Journal — Menu")
        print("1) Record audio from mic (needs speech_recognition & pyaudio)")
        print("2) Transcribe an existing audio file (wav)")
        print("3) Write a text entry manually")
        print("4) Show last entries")
        print("5) Visualize mood trend (last 50)")
        print("6) Exit")
        choice = input("Choose: ").strip()
        if choice == "1":
            if record_to_file is None:
                print("Audio recording not available in this environment. Install speech_recognition & pyaudio.")
                continue
            filename = input("Filename (entry.wav): ").strip() or "entry.wav"
            dur = input("Duration seconds (default 10): ").strip()
            try:
                dur = int(dur) if dur else 10
            except:
                dur = 10
            record_to_file(filename, duration=dur)
            text = transcribe_file(filename)
            create_entry_from_text(text)
        elif choice == "2":
            fn = input("Path to wav file: ").strip()
            if not fn:
                print("No file provided.")
                continue
            if transcribe_file is None:
                print("Transcription not available: install speech_recognition.")
                continue
            text = transcribe_file(fn)
            create_entry_from_text(text)
        elif choice == "3":
            print("Write your entry. End with an empty line.")
            lines = []
            while True:
                l = input()
                if l.strip() == "":
                    break
                lines.append(l)
            text = "\n".join(lines).strip()
            if not text:
                print("Empty entry.")
                continue
            create_entry_from_text(text)
        elif choice == "4":
            entries = load_entries(limit=10)
            for e in entries[::-1]:
                print("-"*40)
                print(e["timestamp"], "|", e["sentiment"].get("label"))
                print(e["summary"])
        elif choice == "5":
            entries = load_entries(limit=200)
            plot_sentiment(entries, show=True)
        elif choice == "6":
            print("Goodbye.")
            sys.exit(0)
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    Path("entries.jsonl").touch(exist_ok=True)
    cli_menu()
