# 3️⃣ Move Zeroes

# Problem Statement:
# Given an integer array nums, move all 0s to the end while maintaining the relative order of non-zero elements.
# You must do this in-place.

# Example:

# Input: nums = [0,1,0,3,12]  
# Output: [1,3,12,0,0]


# Python Code:

def moveZeroes(nums):
    left = 0
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1