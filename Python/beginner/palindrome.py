#!/usr/bin/env python3
"""Palindrome check (beginner-friendly)

This script checks whether a given string or number is a palindrome.
It normalizes input by removing spaces and lowercasing characters so that
phrases like "A man a plan a canal Panama" are detected as palindromes.
"""

def is_palindrome(value) -> bool:
    """Return True if value is a palindrome.

    Converts the input to string, removes spaces and compares to its reverse.
    """
    s = str(value).replace(" ", "").lower()
    return s == s[::-1]


if __name__ == "__main__":
    val = input("Enter a string or number to check for palindrome: ").strip()
    if is_palindrome(val):
        print(f'"{val}" is a palindrome.')
    else:
        print(f'"{val}" is not a palindrome.')
