# Longest Substring Without Repeating Characters

# Problem Statement:
# Given a string s, find the length of the longest substring without repeating characters.

# Example:

# Input: s = "abcabcbb"  
# Output: 3  # ("abc")


# Python Code:

def lengthOfLongestSubstring(s):
    seen = set()
    left = 0
    max_len = 0
    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left])
            left += 1
        seen.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len