# temperature_converter.py

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

def celsius_to_kelvin(celsius):
    return celsius + 273.15

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

print("Temperature Converter")
print("1. Celsius to Fahrenheit")
print("2. Fahrenheit to Celsius")
print("3. Celsius to Kelvin")
print("4. Kelvin to Celsius")

choice = input("Enter your choice (1-4): ")

if choice == '1':
    c = float(input("Enter temperature in Celsius: "))
    print(f"{c}°C = {celsius_to_fahrenheit(c):.2f}°F")

elif choice == '2':
    f = float(input("Enter temperature in Fahrenheit: "))
    print(f"{f}°F = {fahrenheit_to_celsius(f):.2f}°C")

elif choice == '3':
    c = float(input("Enter temperature in Celsius: "))
    print(f"{c}°C = {celsius_to_kelvin(c):.2f}K")

elif choice == '4':
    k = float(input("Enter temperature in Kelvin: "))
    print(f"{k}K = {kelvin_to_celsius(k):.2f}°C")

else:
    print("Invalid choice! Please select between 1 and 4.")
