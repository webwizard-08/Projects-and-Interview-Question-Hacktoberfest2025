def solve_polycarp_coins(n):
    """
    Solve the Polycarp coins distribution problem.
    
    Args:
        n (int): Total sum to be distributed using coins of value 1 and 2
    
    Returns:
        tuple: Number of coins of value 1 and 2 respectively
    """
    a = n // 3  # Base distribution
    b = n % 3   # Remainder to be distributed
    
    if b == 0:
        return (a, a)
    elif b == 1:
        return (a + 1, a)
    else:  # b == 2
        return (a + b % 2, a + b // 2)

# Process multiple test cases
t = int(input())
for _ in range(t):
    n = int(input())
    c1, c2 = solve_polycarp_coins(n)
    print(c1, c2)