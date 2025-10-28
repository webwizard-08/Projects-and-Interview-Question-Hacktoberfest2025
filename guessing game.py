import random

def number_guessing_game():
    number_to_guess = random.randint(1, 100)
    max_attempts = 7
    attempts = 0

    print("I'm thinking of a number between 1 and 100.")
    print(f"You have {max_attempts} attempts to guess it!")

    while attempts < max_attempts:
        try:
            guess = int(input(f"Attempt {attempts + 1}: Enter your guess: "))
            attempts += 1
            if guess < number_to_guess:
                print("Too low! Try again.")
            elif guess > number_to_guess:
                print("Too high! Try again.")
            else:
                print(f"Correct! You guessed the number in {attempts} attempts.")
                break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    else:
        print(f"Sorry, you've used all attempts. The number was {number_to_guess}.")

if __name__ == "__main__":
    number_guessing_game()
