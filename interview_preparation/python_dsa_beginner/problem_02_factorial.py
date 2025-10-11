def factorial(n):
    """
    Compute factorial of a non-negative integer using an iterative approach.
    
    Constraint:
    - n must be a non-negative integer (n >= 0)
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


# Example
print(factorial(5))  # Output: 120
