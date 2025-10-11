# Password Strength Checker
# Author: Lavanya 

import re
def check_password_strength(password):
    strength = 0
    suggestions = []

    # Check length
    if len(password) >= 12:
        strength += 2
    elif len(password) >= 8:
        strength += 1
    else:
        suggestions.append("Use at least 8 characters.")

    # Check lowercase
    if re.search("[a-z]", password):
        strength += 1
    else:
        suggestions.append("Add lowercase letters.")

    # Check uppercase
    if re.search("[A-Z]", password):
        strength += 1
    else:
        suggestions.append("Add uppercase letters.")

    # Check numbers
    if re.search("[0-9]", password):
        strength += 1
    else:
        suggestions.append("Add numbers (0-9).")

    # Check special characters
    if re.search("[!@#$%^&*(),.?\":{}|<>]", password):
        strength += 1
    else:
        suggestions.append("Add special characters (!@#$ etc).")

    # Decide strength level
    if strength >= 5:
        level = "Very Strong"
    elif strength >= 4:
        level = "Strong"
    elif strength >= 3:
        level = "Moderate"
    else:
        level = "Weak"

    return level, suggestions


# Main Program
if __name__ == "__main__":
    pwd = input("Enter a password to check: ")
    level, suggestions = check_password_strength(pwd)

    print(f"\nPassword Strength: {level}")
    if suggestions:
        print("Suggestions to improve:")
        for s in suggestions:
            print(" -", s)
    else:
        print("Your password looks good!")
