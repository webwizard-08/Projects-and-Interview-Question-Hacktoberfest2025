# basic_programs.py
# Author: <your-name>
# Description: Basic Python programs for beginners

# 1. Find factorial
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

# 2. Check prime
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

# 3. Reverse a string
def reverse_string(s):
    return s[::-1]

if __name__ == "__main__":
    print("Factorial of 5:", factorial(5))
    print("Is 7 prime?", is_prime(7))
    print("Reverse of 'Python':", reverse_string('Python'))
