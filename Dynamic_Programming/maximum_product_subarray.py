"""
Maximum Product Subarray Problem
Time Complexity: O(n)
Space Complexity: O(1)

Problem Statement:
Given an integer array nums, find the contiguous subarray (containing at least one number)
which has the largest product, and return the product.

Examples:
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product = 6.

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a contiguous subarray that exists in nums.

Input: nums = [-2,3,-4]
Output: 24
Explanation: Subarray [3,-4,-2] or [-2,3,-4] gives product 24.
"""

def max_product_subarray(nums):
    if not nums:
        return 0

    # Initialize max and min products so far with first element
    max_so_far = nums[0]
    min_so_far = nums[0]
    result = nums[0]

    # Iterate through the array
    for i in range(1, len(nums)):
        num = nums[i]

        # If the current number is negative, swap max and min
        # because multiplying by a negative flips signs
        if num < 0:
            max_so_far, min_so_far = min_so_far, max_so_far

        # Compute max/min product including current element
        max_so_far = max(num, num * max_so_far)
        min_so_far = min(num, num * min_so_far)

        # Update the final result
        result = max(result, max_so_far)

    return result


# Test cases
if __name__ == "__main__":
    tests = [
        [2, 3, -2, 4],
        [-2, 0, -1],
        [-2, 3, -4],
        [0, 2],
        [-2],
        [2, -5, -2, -4, 3]
    ]

    for i, nums in enumerate(tests, 1):
        print(f"Test Case {i}:")
        print(f"Input Array: {nums}")
        print(f"Maximum Product Subarray: {max_product_subarray(nums)}\n")
