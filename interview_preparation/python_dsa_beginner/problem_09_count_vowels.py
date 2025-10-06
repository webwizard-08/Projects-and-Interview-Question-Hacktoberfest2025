def count_vowels(s):
    """
    Count the number of vowels in a string.
    
    Constraint:
    - s must be a string
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count


# Example
print(count_vowels("Hello World"))  # Output: 3
