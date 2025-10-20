"""
visualizer.py
Plot sentiment polarity over time using matplotlib.
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def plot_sentiment(entries, show=True, save_path=None):
    if not entries:
        print("No entries to plot.")
        return
    times = [datetime.fromisoformat(e["timestamp"].replace("Z","")) for e in entries]
    polarities = [e["sentiment"].get("polarity", 0) for e in entries]
    # Sort by time
    times, polarities = zip(*sorted(zip(times, polarities)))
    plt.figure(figsize=(8,4))
    plt.plot(times, polarities, marker='o')
    plt.axhline(0, linestyle='--', linewidth=0.8)
    plt.title("Mood trend (polarity over time)")
    plt.ylabel("Polarity (-1 negative → +1 positive)")
    plt.xlabel("Time")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print("Saved plot to", save_path)
    if show:
        plt.show()
