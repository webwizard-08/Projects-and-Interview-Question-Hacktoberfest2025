t = int(input())  # number of test cases

for _ in range(t):
    a, b, c, d = map(int, input().split())
    
    # To form a square, all four sides must be equal
    if a == b == c == d:
        print("YES")
    else:
        print("NO")
