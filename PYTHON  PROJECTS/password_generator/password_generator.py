# password_generator.py
# Author: <your-name>
# Description: Generates a strong random password using Python.

import random
import string

def generate_password(length=12):
    """Generate a secure random password."""
    characters = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.choice(characters) for _ in range(length))
    return password

if __name__ == "__main__":
    print("🔐 Generated Password:", generate_password(16))
