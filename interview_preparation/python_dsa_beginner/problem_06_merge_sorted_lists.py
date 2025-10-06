def merge_sorted_lists(list1, list2):
    """
    Merge two sorted lists into a single sorted list.
    
    Constraints:
    - list1 and list2 must be sorted lists of integers
    
    Time Complexity: O(n + m)  # n = len(list1), m = len(list2)
    Space Complexity: O(n + m)
    """
    merged = []
    i = j = 0
    
    # Merge elements from both lists
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1
    
    # Add remaining elements
    while i < len(list1):
        merged.append(list1[i])
        i += 1
    while j < len(list2):
        merged.append(list2[j])
        j += 1
    
    return merged


# Example
print(merge_sorted_lists([1, 3, 5], [2, 4, 6]))  # Output: [1, 2, 3, 4, 5, 6]
