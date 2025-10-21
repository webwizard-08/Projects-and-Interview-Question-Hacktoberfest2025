# ==================================================
#                LeetCode: 20
#              Valid Parentheses
# ==================================================
"""
Problem Statement:
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
An input string is valid if:
Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
"""

# Link: https://leetcode.com/problems/valid-parentheses/description/

"""
Constraints:
1 <= s.length <= 104
s consists of parentheses only '()[]{}'.                       
"""

#Explanation:
# Push opening brackets.
# Pop when correct closing found.
# Valid if stack is empty at the end.

#Solution:
def isValid(s):
    stack, mapping = [], {')':'(', '}':'{', ']':'['}
    for c in s:
        if c in mapping:
            if not stack or stack[-1] != mapping[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return not stack
"""
Complexiety Analysis:
Time: O(n)
Space: O(n)
"""

"""
Example 1:
Input: s = "()"
Output: true

Example 2:
Input: s = "()[]{}"
Output: true

Example 3:
Input: s = "(]"
Output: false

Example 4:
Input: s = "([])"
Output: true

Example 5:
Input: s = "([)]"
Output: false
"""