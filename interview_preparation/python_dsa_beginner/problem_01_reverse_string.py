def reverse_string(s):
    """
    Reverse a given string.
    
    Constraint:
    - s must be a string
    
    Time Complexity: O(n)
    Space Complexity: O(n)  # new string is created
    """
    reversed_s = ""
    for char in s:
        reversed_s = char + reversed_s
    return reversed_s


# Example
print(reverse_string("hello"))  # Output: "olleh"
