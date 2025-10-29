import java.util.Scanner;

public class BasicArithmeticCalculator {
    public static void main(String[] args) {
        // Create Scanner object to take user input
        Scanner scanner = new Scanner(System.in);
    
        System.out.println("   Welcome to Basic Arithmetic Operations Calculator!");
        System.out.println("===================================");
        System.out.println();
        
        // Input the first number
        System.out.print("Enter first number: ");
        float firstNum = scanner.nextFloat();

        // Display operator choices
        System.out.println("Choose from:");
        System.out.println("For + enter 1");
        System.out.println("For - enter 2");
        System.out.println("For * enter 3");
        System.out.println("For / enter 4");
        System.out.println("For % enter 5");
        System.out.print("Enter your choice: ");
        int opr = scanner.nextInt(); // Input your operator choice

        // Input the second number
        System.out.print("Enter second number: ");
        float secondNum = scanner.nextFloat(); // Input second number

        // Perform calculation based on user's choice
        switch (opr) {
            case 1:
                // Returns the result of Addition of two numbers.
                System.out.println("Addition of " + firstNum + " and " + secondNum + " is: " + (firstNum + secondNum));
                break;

            case 2:
                // Returns the result of Subtraction of two numbers.
                System.out.println("Subtraction of " + firstNum + " and " + secondNum + " is: " + (firstNum - secondNum));
                break;

            case 3:
                // Returns the result of Multiplication of two numbers.
                System.out.println("Multiplication of " + firstNum + " and " + secondNum + " is: " + (firstNum * secondNum));
                break;

            case 4:
                // Check for division by zero before performing division
                if (secondNum == 0) {
                    System.out.println("Division is not possible (cannot divide by zero)");
                } else {
                    // Returns the result of Division of two numbers.
                    System.out.println("Division of " + firstNum + " and " + secondNum + " is: " + (firstNum / secondNum));
                }
                break;

            case 5:
                // Returns the result of Remainder of two numbers.
                System.out.println("Remainder of " + firstNum + " and " + secondNum + " is: " + (firstNum % secondNum));
                break;

            default:
                // Runs if user made an invalid choice
                System.out.println("Invalid Choice");
        }

        // Close the scanner to prevent memory leaks
        scanner.close();
    }
}