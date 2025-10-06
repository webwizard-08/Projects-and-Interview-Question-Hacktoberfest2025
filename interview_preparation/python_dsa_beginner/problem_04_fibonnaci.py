def fibonacci(n):
    """
    Compute the nth Fibonacci number using an iterative approach.
    
    Constraint:
    - n must be a non-negative integer (n >= 0)
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# Example
print(fibonacci(7))  # Output: 13
