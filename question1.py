import sys
input = sys.stdin.readline

def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        result = 0
        # For each subarray, the optimal partition is to split when a new minimum occurs.
        # For this problem, f(b) is number of pieces (splits at new minimums).
        # We use a monotonic stack to efficiently compute contributions.
        stack = []
        add = [0] * n
        for i, v in enumerate(a):
            while stack and a[stack[-1]] > v:
                stack.pop()
            j = stack[-1] if stack else -1
            add[i] = i - j
            stack.append(i)
        # Now calculate result using the number of subarrays ending at i where a[i] is the min
        for i in range(n):
            result += a[i] * add[i]
        print(result)
        
if __name__ == "__main__":
    solve()
