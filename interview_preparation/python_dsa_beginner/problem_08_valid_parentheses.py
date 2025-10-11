def is_valid_parentheses(s):
    """
    Check if the string has valid parentheses.
    
    Constraint:
    - s must be a string containing only '()', '{}', and '[]'
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping.values():  # opening brackets
            stack.append(char)
        elif char in mapping:  # closing brackets
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
    return not stack


# Example
print(is_valid_parentheses("()[]{}"))  # Output: True
print(is_valid_parentheses("(]"))      # Output: False
