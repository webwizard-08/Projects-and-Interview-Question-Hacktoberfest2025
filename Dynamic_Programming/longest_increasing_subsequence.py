"""
Longest Increasing Subsequence (LIS) Problem Solution
Time Complexity: O(n^2)
Space Complexity: O(n)

Problem Statement:
Given an array of integers, find the length of the longest subsequence such that 
all elements of the subsequence are sorted in increasing order.

A subsequence is a sequence that can be derived from an array by deleting some 
or no elements without changing the order of the remaining elements.

Example:
nums = [10, 9, 2, 5, 3, 7, 101, 18]
LIS length: 4
One possible LIS: [2, 3, 7, 101]
"""

def longest_increasing_subsequence(nums):
    n = len(nums)
    if n == 0:
        return 0
    
    # dp[i] stores the length of LIS ending at index i
    dp = [1] * n  
    
    # Build the dp table
    for i in range(1, n):
        for j in range(0, i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def print_lis(nums):
    n = len(nums)
    if n == 0:
        return []
    
    dp = [1] * n
    prev = [-1] * n  # To reconstruct the LIS path
    
    # Fill dp and prev arrays
    for i in range(1, n):
        for j in range(0, i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j  # Store the index of the previous element in LIS
    
    # Find index of the maximum LIS length
    lis_length = max(dp)
    lis_end = dp.index(lis_length)
    
    # Reconstruct LIS sequence
    lis = []
    while lis_end != -1:
        lis.append(nums[lis_end])
        lis_end = prev[lis_end]
    
    return list(reversed(lis))

# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [10, 9, 2, 5, 3, 7, 101, 18]
    print("Test Case 1:")
    print(f"Input Array: {nums1}")
    print(f"Length of LIS: {longest_increasing_subsequence(nums1)}")
    print(f"LIS: {print_lis(nums1)}\n")
    
    # Test case 2
    nums2 = [0, 1, 0, 3, 2, 3]
    print("Test Case 2:")
    print(f"Input Array: {nums2}")
    print(f"Length of LIS: {longest_increasing_subsequence(nums2)}")
    print(f"LIS: {print_lis(nums2)}\n")
    
    # Test case 3
    nums3 = [7, 7, 7, 7, 7, 7, 7]
    print("Test Case 3:")
    print(f"Input Array: {nums3}")
    print(f"Length of LIS: {longest_increasing_subsequence(nums3)}")
    print(f"LIS: {print_lis(nums3)}")
