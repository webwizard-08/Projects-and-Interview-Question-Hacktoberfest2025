def min_decreases_to_make_awesome(test_cases):
    res = []
    for n, a in test_cases:
        cost = 0
        a = a[:]  # Make a copy
        for i in range(n - 2, -1, -1):
            # Odd: a[i] < a[i+1], Even: a[i] > a[i+1]
            # Since i is 0-based, position (i+1) is 1-based
            if (i % 2 == 0 and a[i] >= a[i+1]):
                # Make a[i] < a[i+1]
                to_dec = a[i] - (a[i+1] - 1)
                cost += to_dec
                a[i] -= to_dec
            elif (i % 2 == 1 and a[i] <= a[i+1]):
                # Make a[i] > a[i+1]
                to_dec = (a[i+1] + 1) - a[i]
                cost += to_dec
                a[i] += to_dec
        res.append(cost)
    return res

# Example usage
t = 7
inputs = [
    (5, [1, 4, 2, 5, 3]),
    (4, [3, 3, 2, 1]),
    (5, [6, 6, 6, 6, 6]),
    (7, [1, 2, 3, 4, 5, 6, 7]),
    (3, [3, 2, 1]),
    (2, [1, 2]),
    (9, [65, 85, 19, 53, 21, 79, 92, 29, 96])
]
ans = min_decreases_to_make_awesome(inputs)
print('\n'.join(map(str, ans)))
