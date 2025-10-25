"""
Distinct Subsequences Problem
Time Complexity: O(m * n)
Space Complexity: O(m * n)

Problem Statement:
Given two strings s and t, return the number of distinct subsequences of s 
which equal t.

A subsequence of a string is a new string formed from the original string by 
deleting some (can be none) of the characters without disturbing the relative 
positions of the remaining characters.

Example:
Input: s = "rabbbit", t = "rabbit"
Output: 3
Explanation: 
There are 3 ways to form "rabbit" from "rabbbit":
1. rabbbit
2. rabbbit
3. rabbbit (different 'b's removed each time)

Input: s = "babgbag", t = "bag"
Output: 5
Explanation:
There are 5 distinct subsequences of s that equal t.
"""

def num_distinct(s, t):
    m, n = len(s), len(t)
    
    # dp[i][j] -> number of distinct subsequences of s[0..i-1] that form t[0..j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # An empty t can always be formed from any prefix of s (by deleting all characters)
    for i in range(m + 1):
        dp[i][0] = 1
    
    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                # Option 1: include s[i-1], Option 2: skip s[i-1]
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
            else:
                # Skip current char in s
                dp[i][j] = dp[i - 1][j]
    
    return dp[m][n]

# Test cases
if __name__ == "__main__":
    tests = [
        ("rabbbit", "rabbit"),
        ("babgbag", "bag"),
        ("abc", "abc"),
        ("abc", "ax"),
        ("aaaaa", "aa")
    ]
    
    for i, (s, t) in enumerate(tests, 1):
        print(f"Test Case {i}:")
        print(f"s = '{s}', t = '{t}'")
        print(f"Number of Distinct Subsequences: {num_distinct(s, t)}\n")
