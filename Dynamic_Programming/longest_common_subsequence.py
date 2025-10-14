"""
Longest Common Subsequence (LCS) Problem Solution
Time Complexity: O(m*n) where m and n are the lengths of the input strings
Space Complexity: O(m*n)

Problem Statement:
Given two strings, find the length of the longest subsequence present in both of them.
A subsequence is a sequence that appears in the same relative order, but not necessarily contiguous.

Example:
str1 = "ABCDGH"
str2 = "AEDFHR"
LCS length: 3 (ADH is the longest common subsequence)
"""

def longest_common_subsequence(str1, str2):
    m = len(str1)
    n = len(str2)
    
    # Create a table to store results of subproblems
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Return the length of LCS
    return dp[m][n]

def print_lcs(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Backtrack to find the LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            lcs.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))

# Test cases
if __name__ == "__main__":
    # Test case 1
    str1 = "ABCDGH"
    str2 = "AEDFHR"
    print(f"Test Case 1:")
    print(f"String 1: {str1}")
    print(f"String 2: {str2}")
    print(f"Length of LCS: {longest_common_subsequence(str1, str2)}")
    print(f"LCS: {print_lcs(str1, str2)}\n")
    
    # Test case 2
    str1 = "AGGTAB"
    str2 = "GXTXAYB"
    print(f"Test Case 2:")
    print(f"String 1: {str1}")
    print(f"String 2: {str2}")
    print(f"Length of LCS: {longest_common_subsequence(str1, str2)}")
    print(f"LCS: {print_lcs(str1, str2)}")
