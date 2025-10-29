import java.util.Scanner;

public class Temperature_Converter {
    public static void main(String[] args) {
        // Create Scanner object to take user input
        Scanner scanner = new Scanner(System.in);
    
        System.out.println("===================================");
        System.out.println("   Welcome to Temperature Converter!");
        System.out.println("===================================");
        System.out.println();

        float temp = 0;  // Variable to store temperature value
        String unit = ""; // Variable to store unit type (celsius, fahrenheit, kelvin)

        // Input Temperature (with error handling)
        try {
            System.out.print("Enter Temperature: ");
            temp = scanner.nextFloat(); // Take numeric input for temperature
        } 
        catch (Exception e) {
            // Display message if invalid temperature entered
            System.out.println("Error: Enter a valid numeric temperature!");
            scanner.close();
            return; // Exit program on invalid input
        }

        scanner.nextLine(); 
        // Input Temperature Unit (with error handling)

        try {
            System.out.print("Enter Your Temperature Unit (celsius/kelvin/fahrenheit): ");
            unit = scanner.nextLine(); // Take unit input
            unit = unit.toLowerCase(); // Convert to lowercase to avoid case mismatch
        } 
        catch (Exception e) {
            // Display message if invalid unit entered
            System.out.println("Error: Enter Valid Unit!");
            scanner.close();
            return; // Exit if error occurs
        }

        switch (unit) {
            case "celsius":
                // Convert Celsius to Fahrenheit and Kelvin
                float f = (temp * 9 / 5) + 32;
                float k = temp + 273.15f;
                System.out.println(temp +" deg C = " + f +" deg F and " + k + "K");
                break;

            case "fahrenheit":
                // Convert Fahrenheit to Celsius and Kelvin
                float c1 = (temp - 32) * 5 / 9;
                float k1 = c1 + 273.15f;
                System.out.println(temp + " deg F = " + c1 + " deg C and " + k1 + "K");
                break;

            case "kelvin":
                // Convert Kelvin to Celsius and Fahrenheit
                float c2 = temp - 273.15f;
                float f2 = (c2 * 9 / 5) + 32;
                System.out.println(temp + "K = " + c2 + " deg C and " + f2 + " deg F");
                break;

            default:
                // Runs if user entered an invalid unit
                System.out.println("Invalid unit entered! Please enter celsius, fahrenheit, or kelvin only.");
                return;
        }

        // Close scanner to prevent memory leak
        scanner.close();
    }
}
