def is_prime(n):
    """
    Check if a number is prime.
    
    Constraint:
    - n must be a non-negative integer (n >= 0)
    
    Time Complexity: O(sqrt(n))
    Space Complexity: O(1)
    """
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


# Example
print(is_prime(11))  # Output: True
print(is_prime(15))  # Output: False
