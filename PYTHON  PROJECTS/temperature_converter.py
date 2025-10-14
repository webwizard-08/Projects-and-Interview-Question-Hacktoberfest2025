# temperature_converter.py
# A robust temperature conversion tool that converts between Celsius, Fahrenheit, and Kelvin.
# Designed for Hacktoberfest contributions with modular design, error handling, and user-friendly interface.

# Conversion Functions
def celsius_to_fahrenheit(celsius):
    # Converts temperature from Celsius to Fahrenheit using the formula: (°C * 9/5) + 32
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    # Converts temperature from Fahrenheit to Celsius using the formula: (°F - 32) * 5/9
    return (fahrenheit - 32) * 5/9

def celsius_to_kelvin(celsius):
    # Converts temperature from Celsius to Kelvin by adding 273.15
    return celsius + 273.15

def kelvin_to_celsius(kelvin):
    # Converts temperature from Kelvin to Celsius by subtracting 273.15
    # Validates that the input is not below absolute zero (0K)
    if kelvin < 0:
        raise ValueError("Temperature cannot be below 0 Kelvin")
    return kelvin - 273.15

# Conversion Configuration
# Dictionary mapping user choices to conversion functions and metadata
conversions = {
    '1': {
        'func': celsius_to_fahrenheit,
        'input_unit': 'Celsius',
        'output_unit': 'Fahrenheit',
        'prompt': 'Enter temperature in Celsius: '
    },
    '2': {
        'func': fahrenheit_to_celsius,
        'input_unit': 'Fahrenheit',
        'output_unit': 'Celsius',
        'prompt': 'Enter temperature in Fahrenheit: '
    },
    '3': {
        'func': celsius_to_kelvin,
        'input_unit': 'Celsius',
        'output_unit': 'Kelvin',
        'prompt': 'Enter temperature in Celsius: '
    },
    '4': {
        'func': kelvin_to_celsius,
        'input_unit': 'Kelvin',
        'output_unit': 'Celsius',
        'prompt': 'Enter temperature in Kelvin: '
    }
}

# User Interface Functions
def display_menu():
    # Displays the conversion menu to the user
    print("\nTemperature Converter")
    print("1. Celsius to Fahrenheit")
    print("2. Fahrenheit to Celsius")
    print("3. Celsius to Kelvin")
    print("4. Kelvin to Celsius")
    print("5. Exit")

def get_valid_choice():
    # Prompts the user for a menu choice and validates it
    # Returns a valid choice (1-5) or prompts again if invalid
    while True:
        choice = input("Enter your choice (1-5): ").strip()
        if choice in conversions or choice == '5':
            return choice
        print("Invalid choice! Please select between 1 and 5.")

def get_valid_temperature(prompt):
    # Prompts the user for a temperature value and ensures it's numeric
    # Returns a float or prompts again if invalid
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input! Please enter a numeric value.")

# Main Program
def main():
    # Main loop to run the temperature converter until the user chooses to exit
    while True:
        display_menu()  # Show the menu
        choice = get_valid_choice()  # Get a valid menu choice

        if choice == '5':  # Exit condition
            print("Exiting program.")
            break

        # Perform the selected conversion
        conversion = conversions[choice]
        try:
            temp = get_valid_temperature(conversion['prompt'])  # Get temperature input
            result = conversion['func'](temp)  # Apply conversion function
            # Display result formatted to 2 decimal places
            print(f"{temp:.2f}°{conversion['input_unit']} = {result:.2f}°{conversion['output_unit']}")
        except ValueError as e:
            print(f"Error: {e}")  # Handle invalid inputs or Kelvin < 0

#Entry Point
if __name__ == "__main__":
    # Ensures the program runs only when executed directly, not when imported
    main()