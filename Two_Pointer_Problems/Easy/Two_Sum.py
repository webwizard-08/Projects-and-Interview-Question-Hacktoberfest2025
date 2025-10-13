# 1️⃣ Two Sum (Sorted Input)

# Problem Statement:
# Given a sorted array of integers nums and an integer target, return the indices (1-based) of the two numbers such that they add up to target.
# You must use only the two-pointer approach.

# Example:

# Input: nums = [2,7,11,15], target = 9  
# Output: [1,2]


# Python Code:

def twoSum(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            return [left + 1, right + 1]
        elif s < target:
            left += 1
        else:
            right -= 1
    return []