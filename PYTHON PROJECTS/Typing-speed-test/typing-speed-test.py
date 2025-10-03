import random
import time

# Sample texts for typing test
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Python is a great programming language for beginners",
    "Typing speed tests are fun and improve your skills",
    "Practice makes perfect when learning to type faster",
    "Artificial intelligence is transforming the world rapidly"
]

def get_random_text():
    return random.choice(texts)

def typing_test():
    print("=== Typing Speed Test ===\n")
    text_to_type = get_random_text()
    print("Type the following text as fast as you can:\n")
    print(f'"{text_to_type}"\n')

    input("Press Enter when you are ready...")
    print("\nStart typing now!\n")

    start_time = time.time()
    user_input = input()
    end_time = time.time()

    time_taken = end_time - start_time
    words = text_to_type.split()
    user_words = user_input.split()

    # Calculate accuracy
    correct_words = 0
    for i in range(min(len(words), len(user_words))):
        if words[i] == user_words[i]:
            correct_words += 1
    accuracy = (correct_words / len(words)) * 100

    # Calculate Words Per Minute (WPM)
    wpm = len(user_words) / (time_taken / 60)

    print("\n=== Results ===")
    print(f"Time Taken: {time_taken:.2f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Words Per Minute (WPM): {wpm:.2f}")

if __name__ == "__main__":
    typing_test()
