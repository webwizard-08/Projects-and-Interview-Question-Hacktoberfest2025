#!/usr/bin/env python3
"""Fibonacci series (beginner-friendly)

This script generates the first n Fibonacci numbers and prints them.
The Fibonacci series starts with 0, 1 and each subsequent number is the sum
of the previous two.
"""

from typing import List


def fibonacci(n: int) -> List[int]:
    """Return a list of the first n Fibonacci numbers.

    Args:
        n: Number of Fibonacci elements to generate (n >= 0).

    Returns:
        A list containing the first n Fibonacci numbers.
    """
    if n <= 0:
        return []
    seq = [0]
    if n == 1:
        return seq
    seq.append(1)
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq


if __name__ == "__main__":
    try:
        n = int(input("How many Fibonacci numbers to generate? ").strip())
        if n <= 0:
            print("Please enter a positive integer.")
        else:
            seq = fibonacci(n)
            print("Fibonacci sequence:", ", ".join(str(x) for x in seq))
    except ValueError:
        print("Please enter a valid integer.")
