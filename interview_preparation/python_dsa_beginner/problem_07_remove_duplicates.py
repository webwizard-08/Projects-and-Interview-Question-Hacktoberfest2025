def remove_duplicates(lst):
    """
    Remove duplicates from a list.
    
    Constraint:
    - lst must be a list of elements (integers, strings, etc.)
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    unique = []
    for item in lst:
        if item not in unique:
            unique.append(item)
    return unique


# Example
print(remove_duplicates([1, 2, 2, 3, 4, 4, 5]))  # Output: [1, 2, 3, 4, 5]
