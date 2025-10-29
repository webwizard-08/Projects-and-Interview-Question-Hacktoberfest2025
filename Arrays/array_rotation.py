def rotate_array(arr, k):
    """
    Rotates an array to the right by k steps.
    
    Args:
        arr (List[int]): Array to rotate
        k (int): Number of positions to rotate right
        
    Returns:
        List[int]: Rotated array
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not arr:
        return arr
        
    n = len(arr)
    k = k % n  # Normalize k to be within array bounds
    
    def reverse(arr, start, end):
        """Helper function to reverse a portion of the array"""
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    
    # Reverse the entire array
    reverse(arr, 0, n - 1)
    # Reverse first k elements
    reverse(arr, 0, k - 1)
    # Reverse remaining elements
    reverse(arr, k, n - 1)
    
    return arr

def rotate_array_slice(arr, k):
    """
    Alternative implementation using Python slicing.
    Less space efficient but more Pythonic.
    
    Args:
        arr (List[int]): Array to rotate
        k (int): Number of positions to rotate right
        
    Returns:
        List[int]: Rotated array
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not arr:
        return arr
        
    k = k % len(arr)
    return arr[-k:] + arr[:-k]

# Test cases
if __name__ == "__main__":
    # Test case 1: Basic rotation
    arr1 = [1, 2, 3, 4, 5, 6, 7]
    k1 = 3
    result1 = rotate_array(arr1.copy(), k1)
    assert result1 == [5, 6, 7, 1, 2, 3, 4]
    assert rotate_array_slice(arr1, k1) == [5, 6, 7, 1, 2, 3, 4]
    
    # Test case 2: Rotation with k > array length
    arr2 = [1, 2, 3]
    k2 = 5
    result2 = rotate_array(arr2.copy(), k2)
    assert result2 == [2, 3, 1]
    assert rotate_array_slice(arr2, k2) == [2, 3, 1]
    
    # Test case 3: Empty array
    assert rotate_array([], 3) == []
    assert rotate_array_slice([], 3) == []
    
    # Test case 4: Single element
    assert rotate_array([1], 2) == [1]
    assert rotate_array_slice([1], 2) == [1]
    
    # Test case 5: Zero rotation
    arr5 = [1, 2, 3, 4]
    assert rotate_array(arr5.copy(), 0) == [1, 2, 3, 4]
    assert rotate_array_slice(arr5, 0) == [1, 2, 3, 4]
    
    print("All array rotation tests passed!")