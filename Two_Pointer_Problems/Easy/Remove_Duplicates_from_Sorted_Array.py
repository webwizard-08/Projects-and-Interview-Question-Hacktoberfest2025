# 2️⃣ Remove Duplicates from Sorted Array

# Problem Statement:
# Given a sorted array nums, remove the duplicates in-place such that each element appears only once and return the new length.
# You must do this using constant extra space.

# Example:

# Input: nums = [1,1,2]  
# Output: 2, nums = [1,2,_]


# Python Code:

def removeDuplicates(nums):
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1