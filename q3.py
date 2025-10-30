# Number of test cases
q = int(input())

for _ in range(q):
    n = int(input())
    s, t = input().split()

    # Sort both strings and compare
    if sorted(s) == sorted(t):
        print("YES")
    else:
        print("NO")
