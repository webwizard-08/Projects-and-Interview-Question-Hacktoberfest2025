def binary_search(arr, target):
    """
    Perform binary search to find the index of target in a sorted array.
    
    Constraints:
    - arr must be a sorted list of integers
    - target must be an integer
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # target not found


# Example
print(binary_search([1, 2, 3, 4, 5, 6], 4))  # Output: 3
print(binary_search([1, 2, 3, 4, 5, 6], 7))  # Output: -1
