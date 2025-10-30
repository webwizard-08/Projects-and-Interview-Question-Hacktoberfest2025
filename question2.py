import sys
import math
input = sys.stdin.readline

def solve():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        a = list(map(int, input().split()))

        # We try all divisors d of a[i] (from 1 to max(a)) in descending order
        # For each d, check if we can get rid of enough numbers not divisible by d, using at most k erases
        # If yes, that's our answer

        maxA = max(a)
        freq = [0] * (maxA + 2)

        for val in a:
            freq[val] += 1

        answer = 1
        # Try all possible d from maxA down to 1
        for d in range(maxA, 0, -1):

            # Count how many numbers in a[] are divisible by d
            cnt = 0
            for mul in range(d, maxA + 1, d):
                cnt += freq[mul]

            # Need to remove (n - cnt) numbers not divisible by d, can we do that in k erases?
            if n - cnt <= k:
                answer = d
