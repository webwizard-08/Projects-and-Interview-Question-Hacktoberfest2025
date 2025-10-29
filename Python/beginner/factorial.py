#!/usr/bin/env python3
"""Factorial (beginner-friendly)

This script shows how to compute the factorial of a non-negative integer.
It includes clear comments and a simple command-line prompt for interactive use.
"""

def factorial(n: int) -> int:
    """Return the factorial of n (n!).

    Args:
        n: A non-negative integer.

    Returns:
        The factorial value as an integer.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative integers.")
    result = 1
    # Multiply result by every integer from 2 up to n
    for i in range(2, n + 1):
        result *= i
    return result


if __name__ == "__main__":
    try:
        num = int(input("Enter a non-negative integer: ").strip())
        print(f"{num}! = {factorial(num)}")
    except ValueError as e:
        print("Invalid input:", e)
