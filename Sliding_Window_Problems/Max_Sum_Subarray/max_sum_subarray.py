"""
Maximum Sum Subarray of Size K - Sliding Window Solution

Problem: Given an array of integers and a number k, find the maximum sum of a subarray of size k.

Time Complexity: O(n) where n is the length of the array
Space Complexity: O(1)

Example:
Input: arr = [1, 4, 2, 10, 2, 3, 1, 0, 20], k = 4
Output: 24
Explanation: Maximum sum subarray of length k=4 is [2, 10, 2, 3] with sum 24
"""

def max_sum_subarray(arr: list, k: int) -> tuple:
    """
    Find the maximum sum subarray of size k.
    
    Args:
        arr: List of integers
        k: Size of subarray
    
    Returns:
        Tuple containing:
        - Maximum sum
        - Start index of maximum sum subarray
        - End index of maximum sum subarray
    
    Raises:
        ValueError: If k is larger than array length
    """
    n = len(arr)
    
    # Input validation
    if k > n:
        raise ValueError("Window size cannot be larger than array length")
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    max_start = 0
    
    # Slide window and keep track of maximum sum
    for i in range(n - k):
        # Remove first element of previous window
        window_sum = window_sum - arr[i]
        # Add last element of current window
        window_sum = window_sum + arr[i + k]
        
        # Update maximum sum if current window sum is larger
        if window_sum > max_sum:
            max_sum = window_sum
            max_start = i + 1
    
    return max_sum, max_start, max_start + k - 1

def print_subarray(arr: list, start: int, end: int) -> None:
    """
    Print the subarray elements between start and end indices.
    
    Args:
        arr: Input array
        start: Start index
        end: End index
    """
    print("Subarray elements:", arr[start:end + 1])

def main():
    """Example usage with test cases."""
    test_cases = [
        ([1, 4, 2, 10, 2, 3, 1, 0, 20], 4),
        ([1, 1, 1, 1, 1], 2),
        ([10, 20, 30, 40], 3),
        ([1], 1),
        ([5, -2, 3, 4, -1, 2, -3, 6], 3)
    ]
    
    for arr, k in test_cases:
        try:
            max_sum, start, end = max_sum_subarray(arr, k)
            print(f"\nArray: {arr}")
            print(f"Window size: {k}")
            print(f"Maximum sum: {max_sum}")
            print_subarray(arr, start, end)
        except ValueError as e:
            print(f"\nError: {e}")
            print(f"Array: {arr}")
            print(f"Window size: {k}")

if __name__ == "__main__":
    main()
