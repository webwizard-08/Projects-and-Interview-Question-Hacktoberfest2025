from typing import List

def merge_sort(arr: List[int]) -> List[int]:
    """
    Implementation of merge sort algorithm.
    
    Args:
        arr (List[int]): Array to sort
        
    Returns:
        List[int]: Sorted array
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr
        
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    """
    Merge helper function for merge sort.
    
    Args:
        left (List[int]): Left sorted array
        right (List[int]): Right sorted array
        
    Returns:
        List[int]: Merged sorted array
    """
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
            
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr: List[int]) -> List[int]:
    """
    Implementation of quick sort algorithm.
    
    Args:
        arr (List[int]): Array to sort
        
    Returns:
        List[int]: Sorted array
        
    Time Complexity: Average O(n log n), Worst O(n²)
    Space Complexity: O(log n) due to recursion
    """
    if len(arr) <= 1:
        return arr
        
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def heap_sort(arr: List[int]) -> List[int]:
    """
    Implementation of heap sort algorithm.
    
    Args:
        arr (List[int]): Array to sort
        
    Returns:
        List[int]: Sorted array
        
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    """
    def heapify(arr: List[int], n: int, i: int):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and arr[left] > arr[largest]:
            largest = left
            
        if right < n and arr[right] > arr[largest]:
            largest = right
            
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
    
    # Build max heap
    for i in range(len(arr) // 2 - 1, -1, -1):
        heapify(arr, len(arr), i)
    
    # Extract elements from heap one by one
    for i in range(len(arr) - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
        
    return arr

# Test cases
if __name__ == "__main__":
    # Test arrays
    test_arrays = [
        [],  # Empty array
        [1],  # Single element
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],  # Random array
        [1, 2, 3, 4, 5],  # Already sorted
        [5, 4, 3, 2, 1],  # Reverse sorted
        [1, 1, 1, 1, 1],  # All same elements
    ]
    
    for arr in test_arrays:
        # Test merge sort
        assert merge_sort(arr.copy()) == sorted(arr)
        
        # Test quick sort
        assert quick_sort(arr.copy()) == sorted(arr)
        
        # Test heap sort
        result = arr.copy()
        assert heap_sort(result) == sorted(arr)
    
    print("All sorting algorithm tests passed!")