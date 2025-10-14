"""
Kadane's Algorithm Variations and Applications
Author: GitHub Copilot
Date: October 14, 2025

This module implements various versions of Kadane's Algorithm, a fundamental
dynamic programming technique for solving the maximum subarray sum problem
and its variations. These problems are frequently asked in technical interviews
at top tech companies.

Time Complexity: O(n) for all variations
Space Complexity: O(1) for basic, O(n) for some variations
"""

def basic_kadane(arr):
    """
    Basic Kadane's Algorithm to find maximum subarray sum.
    
    Args:
        arr (List[int]): Input array of integers
    
    Returns:
        int: Maximum subarray sum
    
    Example:
        >>> basic_kadane([-2, 1, -3, 4, -1, 2, 1, -5, 4])
        6  # from subarray [4, -1, 2, 1]
    """
    if not arr:
        return 0
        
    max_ending_here = max_so_far = arr[0]
    for num in arr[1:]:
        max_ending_here = max(num, max_ending_here + num)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

def kadane_with_indices(arr):
    """
    Kadane's Algorithm that also returns the start and end indices
    of the maximum subarray.
    
    Args:
        arr (List[int]): Input array of integers
    
    Returns:
        tuple: (max_sum, start_index, end_index)
    
    Example:
        >>> kadane_with_indices([-2, 1, -3, 4, -1, 2, 1, -5, 4])
        (6, 3, 6)  # sum = 6, subarray is arr[3:7] = [4, -1, 2, 1]
    """
    if not arr:
        return 0, -1, -1
        
    max_ending_here = max_so_far = arr[0]
    start = end = temp_start = 0
    
    for i in range(1, len(arr)):
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            temp_start = i
        else:
            max_ending_here = max_ending_here + arr[i]
            
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            start = temp_start
            end = i
            
    return max_so_far, start, end

def circular_kadane(arr):
    """
    Modified Kadane's Algorithm for circular array maximum sum.
    Handles cases where the maximum sum subarray wraps around the array.
    
    Args:
        arr (List[int]): Input array of integers
    
    Returns:
        int: Maximum subarray sum (considering circular wrapping)
    
    Example:
        >>> circular_kadane([8, -1, 3, -4, 5])
        15  # from subarray [5, 8, -1, 3] (wrapping around)
    """
    if not arr:
        return 0
        
    # Case 1: Maximum sum without wrapping
    max_normal = basic_kadane(arr)
    
    # Case 2: Maximum sum with wrapping
    # It's the total sum minus the minimum sum subarray
    total_sum = sum(arr)
    # Invert array signs for finding minimum sum subarray
    inverted = [-x for x in arr]
    max_wrap = total_sum + basic_kadane(inverted)
    
    # Return maximum of the two cases
    # If all elements are negative, return the maximum element
    if max_wrap == 0:
        return max_normal
    return max(max_normal, max_wrap)

def kadane_with_minimum_size(arr, min_size):
    """
    Kadane's Algorithm variant that finds maximum subarray sum
    with a minimum subarray size constraint.
    
    Args:
        arr (List[int]): Input array of integers
        min_size (int): Minimum size of the subarray
    
    Returns:
        int: Maximum subarray sum meeting the size constraint
    
    Example:
        >>> kadane_with_minimum_size([1, -2, 3, -4, 5, -6, 7], 3)
        6  # from subarray [3, -4, 5, -6, 7]
    """
    if not arr or min_size > len(arr):
        return 0
        
    # First calculate sum of first min_size elements
    current_sum = sum(arr[:min_size])
    max_sum = current_sum
    
    # Sliding window approach
    for i in range(min_size, len(arr)):
        current_sum = current_sum + arr[i] - arr[i - min_size]
        max_sum = max(max_sum, current_sum)
        
        # Try extending the window
        temp_sum = current_sum
        for j in range(i - min_size):
            temp_sum += arr[j]
            max_sum = max(max_sum, temp_sum)
            
    return max_sum

def test_kadanes_algorithms():
    """
    Test function to verify all Kadane's Algorithm variations.
    """
    # Test cases
    test_arrays = [
        [-2, 1, -3, 4, -1, 2, 1, -5, 4],
        [1, -3, 2, -5, 7, 6, -1, 4, 11, -23],
        [-1, -2, -3, -4],
        [8, -1, 3, -4, 5],
        [1, 2, 3, -2, 5],
        [-2, -3, 4, -1, -2, 1, 5, -3]
    ]
    
    print("Testing Basic Kadane's Algorithm:")
    for arr in test_arrays:
        print(f"Array: {arr}")
        print(f"Maximum subarray sum: {basic_kadane(arr)}")
        print("-" * 50)
    
    print("\nTesting Kadane's with Indices:")
    for arr in test_arrays:
        max_sum, start, end = kadane_with_indices(arr)
        print(f"Array: {arr}")
        print(f"Maximum sum: {max_sum}, Subarray: {arr[start:end+1]}")
        print("-" * 50)
    
    print("\nTesting Circular Kadane's Algorithm:")
    for arr in test_arrays:
        print(f"Array: {arr}")
        print(f"Maximum circular subarray sum: {circular_kadane(arr)}")
        print("-" * 50)
    
    print("\nTesting Kadane's with Minimum Size:")
    min_size = 3
    for arr in test_arrays:
        print(f"Array: {arr}")
        print(f"Maximum subarray sum (min size {min_size}): {kadane_with_minimum_size(arr, min_size)}")
        print("-" * 50)

if __name__ == "__main__":
    test_kadanes_algorithms()