import java.util.Scanner;

public class VotingEligiblityChecker {
    public static void main(String[] args) {
         // Create Scanner object to take user input
        Scanner scanner = new Scanner(System.in);
    
        System.out.println("   Welcome to Voting Eligiblity Checker!");
        System.out.println("===================================");
        System.out.println();
        
        // Input age
        System.out.print("Enter your age: ");
            int age = scanner.nextInt();
  

        // Age Validity
        if(age<1 || age > 110)
        {
            System.out.println("Please enter the valid age");
            return;
        }

        if(age>=18)
        System.out.println("Eligible to vote!");

        if(age<18)
        System.out.println("Not Eligible to vote!");
    }
}
