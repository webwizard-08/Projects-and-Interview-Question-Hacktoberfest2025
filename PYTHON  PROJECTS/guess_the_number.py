import random
import os

HIGHSCORE_FILE = "highscore.txt"

def get_random_number(min_val, max_val):
    return random.randint(min_val, max_val)

def get_user_guess(min_val, max_val):
    while True:
        try:
            guess = int(input(f"Enter your guess ({min_val}-{max_val}): "))
            if min_val <= guess <= max_val:
                return guess
            print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input! Please enter an integer.")

def choose_difficulty():
    print("\nSelect difficulty level:")
    print("1. Easy (1-10, unlimited attempts)")
    print("2. Medium (1-50, max 10 attempts)")
    print("3. Hard (1-100, max 7 attempts)")

    while True:
        choice = input("Enter 1, 2, or 3: ")
        if choice == "1":
            return 1, 10, None
        elif choice == "2":
            return 1, 50, 10
        elif choice == "3":
            return 1, 100, 7
        else:
            print("Invalid choice! Please select 1, 2, or 3.")

def load_highscore():
    if not os.path.exists(HIGHSCORE_FILE):
        return None
    with open(HIGHSCORE_FILE, "r") as file:
        try:
            return int(file.read().strip())
        except ValueError:
            return None

def save_highscore(score):
    with open(HIGHSCORE_FILE, "w") as file:
        file.write(str(score))

def play_guess_the_number():
    print("Welcome to 'Guess the Number'!")

    min_val, max_val, max_attempts = choose_difficulty()
    secret_number = get_random_number(min_val, max_val)
    attempts = 0
    highscore = load_highscore()

    while True:
        guess = get_user_guess(micdn_val, max_val)
        attempts += 1

        if guess < secret_number:
            print("Too low. Try again.")
        elif guess > secret_number:
            print("Too high. Try again.")
        else:
            print(f"Congratulations! You guessed the number {secret_number} in {attempts} attempts.")
            score = max(max_val - attempts + 1, 0)
            print(f"Your score: {score}")
            if highscore is None or score > highscore:
                print("New high score! :)")
                save_highscore(score)
            break

        if max_attempts and attempts >= max_attempts:
            print(f"Sorry, you've reached the maximum attempts ({max_attempts}). The number was {secret_number}.")
            break

    # Ask to play again
    play_again = input("\nDo you want to play again? (y/n): ").strip().lower()
    if play_again == "y":
        play_guess_the_number()
    else:
        print("Thank you for playing. Goodbye!")

if __name__ == "__main__":
    play_guess_the_number()
