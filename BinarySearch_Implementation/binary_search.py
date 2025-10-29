def binary_search(arr, target):
    """
    Performs binary search to find target in a sorted array.
    
    Args:
        arr (List[int]): Sorted array of integers
        target (int): Value to find in the array
    
    Returns:
        int: Index of target if found, -1 if not found
    
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
            
    return -1

def binary_search_recursive(arr, target, left=None, right=None):
    """
    Recursive implementation of binary search.
    
    Args:
        arr (List[int]): Sorted array of integers
        target (int): Value to find in the array
        left (int, optional): Left boundary. Defaults to 0.
        right (int, optional): Right boundary. Defaults to len(arr)-1.
    
    Returns:
        int: Index of target if found, -1 if not found
    """
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1
        
    if left > right:
        return -1
        
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# Test cases
if __name__ == "__main__":
    # Test case 1: Basic test
    arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert binary_search(arr1, 5) == 4
    assert binary_search_recursive(arr1, 5) == 4
    
    # Test case 2: Target not in array
    assert binary_search(arr1, 11) == -1
    assert binary_search_recursive(arr1, 11) == -1
    
    # Test case 3: Target at boundaries
    assert binary_search(arr1, 1) == 0  # First element
    assert binary_search(arr1, 10) == 9  # Last element
    assert binary_search_recursive(arr1, 1) == 0
    assert binary_search_recursive(arr1, 10) == 9
    
    # Test case 4: Empty array
    assert binary_search([], 1) == -1
    assert binary_search_recursive([], 1) == -1
    
    print("All test cases passed!")