
import os
import wave
from datetime import datetime

# Lazy import because these packages are optional and may not be installed in the user's environment.
try:
    import speech_recognition as sr
except Exception:
    sr = None

def record_to_file(filename="entry.wav", duration=10):
    """
    Record from default microphone and save to filename.
    duration in seconds.
    """
    if sr is None:
        raise RuntimeError("speech_recognition not installed. Install requirements or record manually and use transcribe_file().")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"Recording for {duration} seconds. Speak now...")
        audio = r.record(source, duration=duration)
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())
    print(f"Saved recording to {filename}")
    return filename

def transcribe_file(filename):
    """
    Transcribe an audio file using Google Web Speech API via SpeechRecognition (requires internet),
    or pocketsphinx if available offline.
    """
    if sr is None:
        raise RuntimeError("speech_recognition not installed.")
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = r.record(source)
    try:
        print("Transcribing with Google Web Speech API (internet required)...")
        text = r.recognize_google(audio)
        return text
    except Exception as e:
        print("Google transcription failed:", e)
        # try pocketsphinx (offline) if available
        try:
            print("Trying pocketsphinx (offline)...")
            text = r.recognize_sphinx(audio)
            return text
        except Exception as e2:
            raise RuntimeError("All transcription methods failed. Error: " + str(e2))
