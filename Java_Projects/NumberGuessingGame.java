import java.util.Random;
import java.util.Scanner;

public class NumberGuessingGame {
    public static void main(String[] args) {
        // Create Random and Scanner objects
        Random random = new Random();
        Scanner scanner = new Scanner(System.in);
        
        // Generate random number between 1 and 100
        int numberToGuess = random.nextInt(100) + 1;
        int numberOfTries = 0;
        int maxAttempts = 7;
        boolean hasWon = false;
        
        System.out.println("===================================");
        System.out.println("   Welcome to Number Guessing Game!");
        System.out.println("===================================");
        System.out.println("I'm thinking of a number between 1 and 100.");
        System.out.println("You have " + maxAttempts + " attempts to guess it!");
        System.out.println();
        
        // Game loop
        while (numberOfTries < maxAttempts && !hasWon) {
            System.out.print("Attempt " + (numberOfTries + 1) + "/" + maxAttempts + " - Enter your guess: ");
            
            // Validate input
            if (!scanner.hasNextInt()) {
                System.out.println("Please enter a valid number!");
                scanner.next(); // Clear invalid input
                continue;
            }
            
            int guess = scanner.nextInt();
            numberOfTries++;
            
            // Check guess
            if (guess < 1 || guess > 100) {
                System.out.println("Please guess a number between 1 and 100!");
                numberOfTries--; // Don't count invalid attempts
            } else if (guess < numberToGuess) {
                System.out.println("Too low! Try again.");
            } else if (guess > numberToGuess) {
                System.out.println("Too high! Try again.");
            } else {
                hasWon = true;
                System.out.println();
                System.out.println("🎉 Congratulations! You guessed it right!");
                System.out.println("The number was: " + numberToGuess);
                System.out.println("You took " + numberOfTries + " attempts.");
            }
            System.out.println();
        }
        
        // Game over message
        if (!hasWon) {
            System.out.println("😢 Game Over! You've used all attempts.");
            System.out.println("The correct number was: " + numberToGuess);
        }
        
        scanner.close();
    }
}
