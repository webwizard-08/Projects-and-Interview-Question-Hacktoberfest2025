"""
Longest Valid Parentheses Problem
Time Complexity: O(n)
Space Complexity: O(n)

Problem Statement:
Given a string containing just the characters '(' and ')',
return the length of the longest valid (well-formed) parentheses substring.

Examples:
Input: s = "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()"

Input: s = ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()"

Input: s = ""
Output: 0
Explanation: No valid parentheses substring exists.
"""

def longest_valid_parentheses(s):
    # Stack to store indices
    stack = [-1]
    max_length = 0

    for i, ch in enumerate(s):
        if ch == '(':
            # Push index of '('
            stack.append(i)
        else:
            # Pop previous '(' index
            stack.pop()
            if not stack:
                # Reset base index when no matching '('
                stack.append(i)
            else:
                # Calculate length of valid substring
                max_length = max(max_length, i - stack[-1])

    return max_length


# Test cases
if __name__ == "__main__":
    tests = [
        "(()",
        ")()())",
        "",
        "()(()",
        "()(())",
        "(((((",
        "()()",
    ]

    for i, s in enumerate(tests, 1):
        print(f"Test Case {i}:")
        print(f"Input String: \"{s}\"")
        print(f"Longest Valid Parentheses Length: {longest_valid_parentheses(s)}\n")
