"""
Wildcard Pattern Matching Problem
Time Complexity: O(m * n)
Space Complexity: O(m * n)

Problem Statement:
Given an input string (s) and a pattern (p), implement wildcard pattern matching 
with support for '?' and '*' where:

'?' -> Matches any single character.
'*' -> Matches any sequence of characters (including the empty sequence).

The matching should cover the entire input string (not partial).

Examples:
Input: s = "aa", p = "a"
Output: False
Explanation: 'a' does not match the entire string "aa"

Input: s = "aa", p = "*"
Output: True
Explanation: '*' can match any sequence

Input: s = "cb", p = "?a"
Output: False
Explanation: '?' matches 'c', but 'a' != 'b'

Input: s = "adceb", p = "*a*b"
Output: True
Explanation: '*' matches 'adce', then 'b' matches 'b'
"""

def is_match(s, p):
    m, n = len(s), len(p)
    
    # dp[i][j] -> True if first i chars of s match first j chars of p
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Empty string matches with empty pattern
    dp[0][0] = True
    
    # Handle patterns with leading '*' (can match empty string)
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == s[i - 1] or p[j - 1] == '?':
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == '*':
                # '*' can match empty (dp[i][j-1]) or one more char (dp[i-1][j])
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
    
    return dp[m][n]

# Test cases
if __name__ == "__main__":
    tests = [
        ("aa", "a"),
        ("aa", "*"),
        ("cb", "?a"),
        ("adceb", "*a*b"),
        ("acdcb", "a*c?b"),
        ("", "*"),
        ("", "?")
    ]
    
    for i, (s, p) in enumerate(tests, 1):
        print(f"Test Case {i}:")
        print(f"Input String: '{s}', Pattern: '{p}'")
        print(f"Match Result: {is_match(s, p)}\n")
